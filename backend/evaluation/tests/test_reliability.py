import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from google.genai.errors import ClientError

from backend.evaluation.dataset import load_dataset
from backend.evaluation.evaluator import ROOT, evaluate
from backend.evaluation.pipeline import PipelineFailure, run_question
from backend.evaluation.reliability import Reliability, ReliabilityConfig, diagnostic
from backend.evaluation.report import aggregate


def error(status, message='private api_key=SECRET authorization=Bearer SECRET cookie=SECRET', retry_after=None):
    exc = ClientError(status, {'error': {'code': status, 'message': message,
                                       'status': 'RESOURCE_EXHAUSTED' if status == 429 else 'INVALID_ARGUMENT'}})
    if retry_after is not None:
        exc.response = SimpleNamespace(headers={'Retry-After': retry_after, 'Authorization': 'SECRET'}, status_code=status)
    return exc


def test_diagnostics_never_serialize_raw_provider_data():
    exc = error(429,retry_after='75')
    exc.details['error']['arbitrary'] = 'SECRET'
    result = diagnostic(exc, 'generation')
    assert result['http_status'] == 429
    assert result['retry_after_seconds'] == 75
    assert result['category'] == 'rate_or_quota'
    assert result['exception_module'] == 'google.genai.errors'
    assert 'SECRET' not in json.dumps(result)
    assert 'Authorization' not in json.dumps(result)


def test_instructor_nested_cause_is_preserved_safely():
    from instructor.core.exceptions import InstructorRetryException, FailedAttempt
    root = error(429)
    exc = InstructorRetryException('SECRET', n_attempts=1, total_usage=0,
                                    messages=[{'content':'SECRET'}], create_kwargs={'api_key':'SECRET'},
                                    failed_attempts=[FailedAttempt(1,root,{'secret':'SECRET'})])
    result = diagnostic(exc,'ragas_faithfulness')
    assert result['instructor_retry_exhausted']
    assert result['category'] == 'rate_or_quota'
    assert result['root_exception_class'] == 'ClientError'
    assert 'SECRET' not in json.dumps(result)


@pytest.mark.parametrize(('exc','category','retryable'), [
    (error(429),'rate_or_quota',True),
    (error(503),'capacity',True),
    (TimeoutError('SECRET'),'timeout',True),
    (error(400),'invalid_request',False),
    (error(403),'authentication_or_permission',False),
    (ValueError('SECRET'),'unknown',False),
    (json.JSONDecodeError('SECRET','SECRET',0),'structured_output_validation',False),
])
def test_classification(exc,category,retryable):
    d=diagnostic(exc,'test')
    assert d['category']==category
    assert d['retryable']==retryable
    assert 'SECRET' not in json.dumps(d)


def test_retry_after_http_date():
    from email.utils import format_datetime
    from datetime import datetime, timezone, timedelta
    d=diagnostic(error(429,retry_after=format_datetime(datetime.now(timezone.utc)+timedelta(seconds=90))),'test')
    assert 88 <= d['retry_after_seconds'] <= 90


def test_bounded_retries_respect_retry_after():
    waits=[]
    p=Reliability(ReliabilityConfig(0,0,2,backoff_base=1),sleep=waits.append)
    fn=MagicMock(side_effect=error(429,retry_after='75'))
    with pytest.raises(ClientError): p.call(fn,'generation')
    assert fn.call_count==3
    assert waits==[75,75]
    assert len(p.events)==3


def test_large_retry_after_is_not_truncated_into_an_early_retry():
    p=Reliability(ReliabilityConfig(0,0,2),sleep=MagicMock())
    fn=MagicMock(side_effect=error(429,retry_after='9999'))
    with pytest.raises(ClientError): p.call(fn,'generation')
    assert fn.call_count==1
    p.sleep.assert_not_called()
    assert p.events[0]['retry_suppressed']=='retry_after_exceeds_run_wait_limit'


def test_non_retryable_calls_once():
    p=Reliability(ReliabilityConfig(0,0,2),sleep=MagicMock())
    fn=MagicMock(side_effect=error(400))
    with pytest.raises(ClientError): p.call(fn,'generation')
    assert fn.call_count==1
    p.sleep.assert_not_called()


def test_shared_pacing_and_async_retry():
    now=[0.0]; waits=[]
    def sleep(delay):
        waits.append(delay);now[0]+=delay
    async def asleep(delay): sleep(delay)
    p=Reliability(ReliabilityConfig(6,6,1,backoff_base=10),clock=lambda:now[0],sleep=sleep,asleep=asleep)
    assert p.call(lambda:'answer','generation','generation')=='answer'
    attempts=[0]
    async def judge():
        attempts[0]+=1
        if attempts[0]==1: raise error(503)
        return .8
    assert asyncio.run(p.acall(judge,'ragas_faithfulness'))==.8
    assert waits==[6,10]
    assert attempts[0]==2


@pytest.mark.parametrize('options',[{'max_retries':-1},{'max_retries':6},{'judge_delay':float('nan')},
                                    {'request_delay':-1},{'max_retries':1.5},{'request_timeout':0}])
def test_config_validation(options):
    with pytest.raises(ValueError): ReliabilityConfig(**options)


def test_retrieval_survives_generation_failure_and_cleanup():
    dataset=load_dataset(ROOT/'dataset-starter.json')
    d=dataset.model_copy(update={'examples':[dataset.examples[0]]})
    def search(**kwargs):
        return [dict(document_id=kwargs['document_id'],owner_id=kwargs['owner_id'],chunk_index=0,text='Evidence',score=.9)]
    p=Reliability(ReliabilityConfig(0,0,0))
    with patch('backend.app.retrieval.vector_store.search_chunks',side_effect=search), patch(
            'backend.app.analysis.answer_generator.client.models.generate_content',side_effect=error(429)), patch(
            'backend.app.retrieval.vector_store.store_chunks'), patch(
            'backend.evaluation.evaluator.wait_for_vectors'), patch(
            'backend.app.retrieval.vector_store.delete_document_vectors') as cleanup:
        r=asyncio.run(evaluate(d,5,reliability=p))
    row=r['rows'][0]
    assert row['status']=='error'
    assert row['diagnostic']['stage']=='generation'
    assert row['retrieved_chunk_ids']==['refund-policy:0']
    assert row['retrieved'][0]['score']==.9
    assert row['retrieval']['recall']==1
    assert r['summary']['retrieval']['recall']==dict(value=1,count=1)
    assert r['summary']['abstention']['abstention_accuracy']['count']==0
    assert cleanup.call_count==2
    assert 'SECRET' not in json.dumps(r)


@pytest.mark.parametrize('stage',['query_embedding','pinecone_retrieval'])
def test_retrieval_stage_identification(stage):
    example=load_dataset(ROOT/'dataset-starter.json').examples[0]
    p=Reliability(ReliabilityConfig(0,0,0))
    with patch('backend.app.retrieval.embeddings.pinecone.inference.embed',side_effect=error(400) if stage=='query_embedding' else None,
               return_value=[{'values':[0.1]}]), patch('backend.app.retrieval.vector_store.index.query',side_effect=error(503)):
        with pytest.raises(PipelineFailure) as info:
            run_question(example,'doc','owner',5,reliability=p)
    assert info.value.stage==stage
    assert 'retrieved' not in info.value.partial


def test_metric_calculations_do_not_depend_on_pacing():
    from backend.evaluation.retrieval_metrics import retrieval_metrics
    now=[0.0]
    def sleep(delay): now[0]+=delay
    p=Reliability(ReliabilityConfig(6,6,0),clock=lambda:now[0],sleep=sleep)
    expected=retrieval_metrics(['x','a'],['a'],5)
    for _ in range(2):
        assert p.call(lambda:retrieval_metrics(['x','a'],['a'],5),'generation','generation')==expected
    assert now[0]==6


def test_production_retry_bindings_are_restored_on_failure():
    from backend.app.analysis import answer_generator
    from backend.app.retrieval import vector_store, embeddings
    saved=(answer_generator.retry_external, vector_store.retry_external, embeddings.retry_external)
    example=load_dataset(ROOT/'dataset-starter.json').examples[0]
    with patch('backend.app.retrieval.embeddings.pinecone.inference.embed',side_effect=error(400)):
        with pytest.raises(PipelineFailure):
            run_question(example,'doc','owner',5)
    assert saved==(answer_generator.retry_external, vector_store.retry_external, embeddings.retry_external)


def test_google_retry_info_and_daily_quota_are_safe():
    exc=error(429)
    exc.details['error']['details']=[{'retryDelay':'35.5s'},
                                    {'violations':[{'quotaId':'RequestsPerDay-SECRET'}]}]
    result=diagnostic(exc,'generation')
    assert result['retry_after_seconds']==35.5
    assert result['quota_scope']=='daily'
    assert not result['retryable']
    assert 'SECRET' not in json.dumps(result)


def test_coverage_distinguishes_unknown_from_ineligible():
    from backend.evaluation.report import metric_coverage
    dataset=load_dataset(ROOT/'dataset-starter.json')
    coverage=metric_coverage(dataset,[])
    assert coverage['retrieval.recall']['eligible']==2
    assert coverage['retrieval.recall']['unavailable']==2
    assert coverage['retrieval.recall']['ineligible']==2
    assert coverage['abstention.abstention_accuracy']['unavailable']==4
    assert coverage['semantic.faithfulness']['unknown_eligibility']==4

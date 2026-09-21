import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.evaluation.dataset import Dataset, load_dataset, prepare_chunks
from backend.evaluation.evaluator import ROOT, evaluate
from backend.evaluation.pipeline import EvaluationScope, run_question, wait_for_vectors
from backend.evaluation.ragas_metrics import NAMES, RagasJudge
from backend.evaluation.report import aggregate, summary, write_report
from backend.evaluation.retrieval_metrics import abstention_metrics, citation_metrics, mean, retrieval_metrics


@pytest.fixture
def dataset():
    return load_dataset(ROOT / "dataset-starter.json")


def test_starter_manifest(dataset):
    assert len(dataset.examples) == 4
    assert len(prepare_chunks(dataset)) == 2
    assert dataset.synthetic


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(examples=[]),
    lambda d: d.update(unexpected=True),
    lambda d: d['examples'].append(d['examples'][0]),
    lambda d: d['documents'].append(d['documents'][0]),
    lambda d: d['examples'][0].update(question='  '),
    lambda d: d['examples'][0].update(document_id='missing'),
    lambda d: d['examples'][0].update(should_answer='true'),
    lambda d: d['examples'][0].update(expected_answer=None),
    lambda d: d['examples'][0].update(relevant_chunk_ids=[]),
    lambda d: d['examples'][0].update(relevant_chunk_ids=['refund-policy:999']),
    lambda d: d['examples'][0].update(relevant_chunk_ids=['privacy-policy:0']),
    lambda d: d['examples'][0].update(relevant_chunk_ids=['refund-policy:0'] * 2),
    lambda d: d['examples'][1].update(expected_answer='fabricated'),
    lambda d: d['examples'][1].update(relevant_chunk_ids=['refund-policy:0']),
])
def test_malformed_dataset(dataset, mutate):
    data = copy.deepcopy(dataset.model_dump())
    mutate(data)
    with pytest.raises(ValidationError):
        Dataset.model_validate(data)


def test_source_drift_rejected(dataset):
    dataset.documents[0].text += ' Changed.'
    with pytest.raises(ValueError, match='manifest changed'):
        prepare_chunks(dataset)


def test_malformed_json(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{broken')
    with pytest.raises(ValidationError):
        load_dataset(path)


@pytest.mark.parametrize(('retrieved','relevant','k','expected'), [
    (['x','a','b'], ['a','b','c'], 3, (2/3,2/3,1/2)),
    (['a'], ['a'], 5, (1,1/5,1)),
    ([], ['a'], 5, (0,0,0)),
    (['x','a'], ['a'], 1, (0,0,0)),
    (['a','a'], ['a'], 2, (1,1/2,1)),
    (['x'], [], 5, (None,None,None)),
])
def test_retrieval_formulas(retrieved,relevant,k,expected):
    result = retrieval_metrics(retrieved,relevant,k)
    assert tuple(result.values()) == expected


@pytest.mark.parametrize('k', [0,-1,True,1.5])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        retrieval_metrics([],[],k)


def test_mrr_and_exclusion():
    values = [retrieval_metrics(ids, relevant, 5)['reciprocal_rank']
              for ids,relevant in [(['x','a'],['a']), ([],['a']), (['x'],[])]]
    assert mean(values) == {'value': .25, 'count': 2}
    assert mean([]) == mean([None]) == {'value': None, 'count': 0}


@pytest.mark.parametrize(('expected','actual','outcome'), [
    (True,True,'true_answer'), (True,False,'false_abstain'),
    (False,True,'false_answer'), (False,False,'true_abstain')])
def test_abstention(expected,actual,outcome):
    result = abstention_metrics(expected,actual)
    assert result['outcome'] == outcome
    assert result['abstention_accuracy'] == (expected == actual)
    assert result['answerable_accuracy'] == (actual if expected else None)
    assert result['unanswerable_accuracy'] == (not actual if not expected else None)


def test_citations():
    result = citation_metrics(['a','x','a'],['a','b'],True)
    assert result == dict(citation_correctness=.5,supporting_citation=True,
                          correct_citation_count=1,incorrect_citation_count=1,missing_citation_count=1)
    assert citation_metrics([],['a'],True)['missing_citation_count'] == 1
    assert citation_metrics([],[],False)['citation_correctness'] is None
    assert citation_metrics(['x'],[],False)['citation_correctness'] == 0
    assert citation_metrics([],['a'],True)['supporting_citation'] is False


def test_aggregate_empty():
    result = aggregate([])
    assert result['retrieval']['mrr'] == dict(value=None,count=0)
    assert result['abstention']['confusion'] == dict(true_answer=0,true_abstain=0,false_answer=0,false_abstain=0)


def test_scope_rejects_foreign_deletion():
    scope = EvaluationScope(['fixture'])
    delete = MagicMock()
    with pytest.raises(ValueError):
        scope.guard('normal-production-document', scope.owner_id)
    with pytest.raises(ValueError):
        scope.guard(scope.documents['fixture'], 'different-owner')
    other = EvaluationScope(['fixture'])
    with pytest.raises(ValueError):
        scope.guard(other.documents['fixture'], other.owner_id)
    scope.attempted.add('policycue-eval-unregistered')
    with pytest.raises(ValueError):
        scope.cleanup(delete)
    delete.assert_not_called()


def test_cleanup_attempts_all_documents():
    scope = EvaluationScope(['a','b'])
    scope.attempted.update(scope.documents.values())
    delete = MagicMock(side_effect=[RuntimeError('secret'),None])
    results = scope.cleanup(delete)
    assert delete.call_count == 2
    assert [r['status'] for r in results] == ['error','delete_requested']
    assert 'secret' not in json.dumps(results)


def test_readiness_success_and_timeout():
    index = MagicMock()
    index.fetch.return_value = {'vectors': {'doc-0': {'metadata': dict(owner_id='owner',document_id='doc',text='text')}}}
    wait_for_vectors(index,'doc','owner',['text'],timeout=0)
    index.fetch.return_value = {'vectors': {}}
    with pytest.raises(TimeoutError):
        wait_for_vectors(index,'doc','owner',['text'],timeout=0)


@pytest.mark.parametrize('abstain', [False,True])
def test_pipeline_matches_production(dataset,abstain):
    from backend.app.analysis.answer_generator import NO_ANSWER_MESSAGE
    from backend.app.retrieval.vector_store import answer_question
    chunks = [dict(document_id='doc',owner_id='owner',chunk_index=0,text='Evidence',score=.9)]
    answer = NO_ANSWER_MESSAGE if abstain else 'Grounded response'
    with patch('backend.app.retrieval.vector_store.search_chunks',return_value=chunks) as search, patch(
            'backend.app.analysis.answer_generator.generate_answer',return_value=answer) as generate:
        actual = run_question(dataset.examples[0],'doc','owner',5)
        expected = answer_question(dataset.examples[0].question,5,'doc','owner')
        assert {key: actual[key] for key in expected} == expected
        assert actual['retrieved'] == chunks
        assert actual['answerable'] == (not abstain)
        assert generate.call_count == search.call_count == 2
        search.assert_called_with(query=dataset.examples[0].question,top_k=5,document_id='doc',owner_id='owner')


def test_pipeline_scope_check(dataset):
    with patch('backend.app.retrieval.vector_store.search_chunks',return_value=[dict(document_id='foreign')]), patch(
            'backend.app.analysis.answer_generator.generate_answer') as generate:
        with pytest.raises(ValueError,match='scope'):
            run_question(dataset.examples[0],'doc','owner',5)
        generate.assert_not_called()


def test_evaluator_and_report(dataset,tmp_path):
    from backend.app.analysis.answer_generator import NO_ANSWER_MESSAGE
    def search(**kwargs):
        return [dict(document_id=kwargs['document_id'], owner_id=kwargs['owner_id'],chunk_index=0,text='Evidence',score=.8)]
    responses = ['Within 30 days.',NO_ANSWER_MESSAGE,'90 days.',NO_ANSWER_MESSAGE]
    with patch('backend.app.retrieval.vector_store.store_chunks') as store, patch(
            'backend.evaluation.evaluator.wait_for_vectors'), patch(
            'backend.app.retrieval.vector_store.search_chunks',side_effect=search), patch(
            'backend.app.analysis.answer_generator.generate_answer',side_effect=responses), patch(
            'backend.app.retrieval.vector_store.delete_document_vectors') as delete:
        report = asyncio.run(evaluate(dataset,5))
    assert store.call_count == delete.call_count == 2
    assert report['status'] == 'ok'
    assert report['summary']['retrieval']['mrr'] == dict(value=1,count=2)
    assert report['summary']['abstention']['confusion']['true_abstain'] == 2
    assert report['summary']['semantic']['faithfulness']['count'] == 0
    assert 'SYNTHETIC' in summary(report)
    path = write_report(report,tmp_path)
    assert json.loads(path.read_text()) == report
    assert path != write_report(report,tmp_path)
    for doc,owner in (call.args for call in delete.call_args_list):
        assert doc in report['document_mapping'].values()
        assert owner == report['owner_id']


def test_partial_ingestion_cleanup(dataset):
    with patch('backend.app.retrieval.vector_store.store_chunks',side_effect=RuntimeError('secret')), patch(
            'backend.app.retrieval.vector_store.delete_document_vectors') as delete:
        report = asyncio.run(evaluate(dataset,5))
    assert report['status'] == 'error'
    assert report['not_run'] == 4
    delete.assert_called_once()
    assert 'secret' not in json.dumps(report)


def test_judge_eligibility_and_errors(dataset):
    judge = object.__new__(RagasJudge)
    judge.metrics = {name: SimpleNamespace(ascore=AsyncMock(return_value=SimpleNamespace(value=.5))) for name in NAMES}
    scores = asyncio.run(judge.score(dataset.examples[1],'fallback',False,['Evidence']))
    assert all(result['status'] == 'ineligible' for result in scores.values())
    scores = asyncio.run(judge.score(dataset.examples[0],'fallback',False,['Evidence']))
    assert scores['context_precision']['status'] == 'ok'
    assert scores['faithfulness']['status'] == 'ineligible'
    judge.metrics['faithfulness'].ascore.side_effect = RuntimeError('secret')
    judge.metrics['answer_relevancy'].ascore.return_value = SimpleNamespace(value=float('nan'))
    scores = asyncio.run(judge.score(dataset.examples[0],'answer',True,['Evidence']))
    assert scores['faithfulness']['value'] is None
    assert scores['faithfulness']['error_type'] == 'RuntimeError'
    assert scores['faithfulness']['diagnostic']['stage'] == 'ragas_faithfulness'
    assert 'secret' not in json.dumps(scores)
    assert scores['context_precision']['status'] == 'ok'
    assert scores['answer_relevancy']['status'] == 'error'
    judge.metrics['context_precision'].ascore.assert_called_with(user_input=dataset.examples[0].question,
        reference=dataset.examples[0].expected_answer,retrieved_contexts=['Evidence'])


def test_mixed_report_aggregation():
    rows = []
    for expected,actual in [(True,True),(True,False),(False,True),(False,False)]:
        rows.append(dict(status='ok',retrieval=retrieval_metrics(['x','a'],['a'] if expected else [],5),
                         abstention=abstention_metrics(expected,actual),
                         citations=citation_metrics(['a'] if actual else [],['a'] if expected else [],expected),
                         semantic={name: dict(value=.75 if actual else None,
                                             status='ok' if actual else 'ineligible') for name in NAMES}))
    rows.append(dict(status='error'))
    result = aggregate(rows)
    assert result['failed'] == 1
    assert result['retrieval']['mrr'] == dict(value=.5,count=2)
    assert result['abstention']['abstention_accuracy'] == dict(value=.5,count=4)
    assert result['abstention']['confusion'] == dict(true_answer=1,true_abstain=1,false_answer=1,false_abstain=1)
    assert result['citations']['supporting_citation'] == dict(value=.5,count=2)
    assert result['semantic']['faithfulness']['count'] == 2
    assert result['semantic']['faithfulness']['statuses']['ineligible'] == 2


def test_evaluator_continues_after_question_failure(dataset):
    def actual(example,doc,owner,k):
        if example.id == 'refund-001':
            raise RuntimeError('private request details')
        return dict(answer='fallback',answerable=False,retrieved=[],sources=[],source_attributions=[],document_found=False)
    with patch('backend.app.retrieval.vector_store.store_chunks'), patch(
            'backend.evaluation.evaluator.wait_for_vectors'), patch(
            'backend.evaluation.evaluator.run_question',side_effect=actual), patch(
            'backend.app.retrieval.vector_store.delete_document_vectors') as delete:
        report = asyncio.run(evaluate(dataset,5))
    assert report['status'] == 'partial'
    assert report['summary']['completed'] == 3
    assert report['summary']['failed'] == 1
    assert report['summary']['retrieval']['recall'] == dict(value=0,count=1)
    assert delete.call_count == 2
    assert 'private request details' not in json.dumps(report)


def test_cleanup_failure_is_reported(dataset):
    with patch('backend.app.retrieval.vector_store.store_chunks',side_effect=RuntimeError()), patch(
            'backend.app.retrieval.vector_store.delete_document_vectors',side_effect=RuntimeError()):
        report = asyncio.run(evaluate(dataset,5))
    assert report['status'] == 'error'
    assert report['cleanup'][0]['status'] == 'error'
    assert '1 errors' in summary(report)


def test_semantic_failure_marks_run_partial(dataset):
    actual = dict(answer='answer',answerable=True,retrieved=[],sources=[],source_attributions=[],document_found=False)
    judge = SimpleNamespace(model='mock-judge',score=AsyncMock(return_value={
        name: dict(value=None,status='error',error_type='TimeoutError') for name in NAMES}))
    with patch('backend.app.retrieval.vector_store.store_chunks'), patch(
            'backend.evaluation.evaluator.wait_for_vectors'), patch(
            'backend.evaluation.evaluator.run_question',return_value=actual), patch(
            'backend.app.retrieval.vector_store.delete_document_vectors'):
        report = asyncio.run(evaluate(dataset,5,judge))
    assert report['status'] == 'partial'
    assert report['summary']['semantic']['faithfulness']['count'] == 0
    assert report['summary']['semantic']['faithfulness']['statuses']['error'] == 4


def test_cli_validate_is_offline(monkeypatch,capsys):
    from backend.evaluation.evaluator import main
    monkeypatch.setattr('sys.argv',['evaluator','--validate-only','--dataset',str(ROOT / 'dataset-starter.json')])
    assert main() == 0
    assert 'Validated policycue-synthetic-starter/1' in capsys.readouterr().out


def test_progress_reports_completed_examples(dataset):
    messages = []
    actual = dict(answer='fallback',answerable=False,retrieved=[],sources=[],source_attributions=[],document_found=False)
    with patch('backend.app.retrieval.vector_store.store_chunks'), patch(
            'backend.evaluation.evaluator.wait_for_vectors'), patch(
            'backend.evaluation.evaluator.run_question',return_value=actual), patch(
            'backend.app.retrieval.vector_store.delete_document_vectors'):
        report = asyncio.run(evaluate(dataset,5,progress=messages.append))
    assert len(messages) == len(report['rows']) == 4
    assert messages[-1] == 'Completed 4/4: privacy-002 (ok)'

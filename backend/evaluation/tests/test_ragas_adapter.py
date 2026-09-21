"""Exercise the real pinned RAGAS + Instructor + Gemini SDK, without network calls."""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.evaluation.dataset import load_dataset
from backend.evaluation.evaluator import ROOT
from backend.evaluation.ragas_metrics import RagasJudge


def test_real_ragas_gemini_adapter():
    pytest.importorskip('ragas')
    from google.genai import types

    # Test-only judge replies, consumed by the real Instructor response parser.
    responses = [
        {'statements': ['Refunds are available within 30 days.']},
        {'statements': [{'statement': 'Refunds are available within 30 days.', 'reason': 'Supported', 'verdict': 1}]},
        *[{'question': 'When are refunds available?', 'noncommittal': 0} for _ in range(3)],
        {'reason': 'Supports the answer', 'verdict': 1},
    ]
    sdk_responses = [types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(text=json.dumps(response))]), finish_reason='STOP')])
        for response in responses]
    settings = SimpleNamespace(gemini_api_key='offline-test-key',gemini_model='gemini-test')
    example = load_dataset(ROOT / 'dataset-starter.json').examples[0]
    async def run():
        with patch('google.genai.models.AsyncModels.generate_content',
                   new=AsyncMock(side_effect=sdk_responses)) as generate, patch(
                'backend.app.retrieval.embeddings.create_query_embedding',return_value=[1.,0.,0.]) as embed:
            judge = RagasJudge(settings)
            try:
                assert judge.metrics['faithfulness'].llm.is_async
                scores = await judge.score(example,'Refunds are available within 30 days.',True,
                                           ['Refund requests are allowed within 30 days.'])
                assert {result['status'] for result in scores.values()} == {'ok'}, scores
                assert scores['faithfulness']['value'] == 1
                assert embed.call_count == 4
                assert generate.await_count == 6
                for call in generate.call_args_list:
                    assert call.kwargs['model'] == 'gemini-test'
            finally:
                await judge.close()
    asyncio.run(run())


@pytest.mark.parametrize('status', [429, 400])
def test_real_instructor_retry_policy(status):
    pytest.importorskip('ragas')
    from google.genai import types
    from google.genai.errors import ClientError
    from pydantic import BaseModel
    from backend.evaluation.reliability import Reliability, ReliabilityConfig, diagnostic

    class Reply(BaseModel):
        ok: bool

    response = types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(parts=[types.Part(text='{"ok":true}')]), finish_reason='STOP')])
    failure = ClientError(status, {'error': {'status':'RESOURCE_EXHAUSTED' if status==429 else 'INVALID_ARGUMENT',
                                            'message':'api_key=SECRET'}})
    waits=[]
    async def asleep(delay): waits.append(delay)
    policy=Reliability(ReliabilityConfig(0,0,2),asleep=asleep)
    async def run():
        with patch('google.genai.models.AsyncModels.generate_content',
                   new=AsyncMock(side_effect=[failure,response])) as generate:
            judge=RagasJudge(SimpleNamespace(gemini_api_key='offline-test-key',gemini_model='gemini-test'),reliability=policy)
            try:
                judge.current_stage='ragas_faithfulness'
                if status==429:
                    result=await judge.metrics['faithfulness'].llm.agenerate('test',Reply)
                    assert result.ok
                    assert generate.await_count==2
                    assert waits==[60]
                else:
                    with pytest.raises(Exception) as info:
                        await judge.metrics['faithfulness'].llm.agenerate('test',Reply)
                    assert diagnostic(info.value,'test')['category']=='invalid_request'
                    assert generate.await_count==1
                    assert waits==[]
                assert policy.events[0]['http_status']==status
                assert policy.events[0]['instructor_retry_exhausted']
                assert 'SECRET' not in json.dumps(policy.events)
            finally:
                await judge.close()
    asyncio.run(run())

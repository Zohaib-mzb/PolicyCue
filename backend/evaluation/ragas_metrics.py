"""Optional RAGAS 0.4.3 collections API; Gemini judge and production embeddings."""
import asyncio
import math
import os
from importlib.metadata import version
from backend.evaluation.reliability import Reliability, ReliabilityConfig, diagnostic

NAMES = ("faithfulness", "answer_relevancy", "context_precision")


class RagasJudge:
    def __init__(self, settings, model: str | None = None, reliability=None):
        self.reliability = reliability or Reliability(ReliabilityConfig(0, 0, 0))
        self.current_stage = "ragas"
        os.environ["RAGAS_DO_NOT_TRACK"] = "true"
        if version("ragas") != "0.4.3":
            raise RuntimeError("Install backend/requirements-evaluation.txt (RAGAS 0.4.3 required)")
        from google import genai
        from google.genai import types
        from ragas.embeddings.base import BaseRagasEmbedding
        import instructor
        from ragas.llms import InstructorLLM
        from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, Faithfulness
        from backend.app.retrieval.embeddings import create_query_embedding
        policy = self.reliability

        class PolicyCueEmbeddings(BaseRagasEmbedding):
            def embed_text(self, text: str, **kwargs) -> list[float]:
                from unittest.mock import patch
                from backend.app.retrieval import embeddings
                with patch.object(embeddings, "retry_external", lambda op: policy.call(
                        op, "ragas_answer_relevancy_embedding")):
                    return create_query_embedding(text)

            async def aembed_text(self, text: str, **kwargs) -> list[float]:
                return await asyncio.to_thread(self.embed_text, text)

            async def aembed_texts(self, texts, **kwargs):
                return [await self.aembed_text(text) for text in texts]

        self.model = model or settings.gemini_model
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(timeout=int(self.reliability.config.request_timeout * 1000),
                                           retry_options=types.HttpRetryOptions(attempts=1)),
        )
        # The factory builds a synchronous Google wrapper in 0.4.3; collections
        # require agenerate. Use the public wrapper with explicit async support.
        judge = self

        class PacedInstructorLLM(InstructorLLM):
            async def agenerate(self, prompt, response_model):
                parent = super().agenerate
                return await judge.reliability.acall(
                    lambda: parent(prompt, response_model), judge.current_stage)

        llm = PacedInstructorLLM(
            client=instructor.from_genai(
                self.client, use_async=True, mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
            ),
            model=self.model, provider="google", temperature=0.0, max_retries=1,
        )
        self.metrics = {
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=PolicyCueEmbeddings()),
            "context_precision": ContextPrecision(llm=llm),
        }

    async def close(self):
        await self.client.aio.aclose()
        self.client.close()

    async def score(self, example, answer: str, answerable: bool, contexts: list[str]) -> dict:
        results = {}
        for name, metric in self.metrics.items():
            kwargs = {"user_input": example.question}
            if name == "context_precision":
                eligible = bool(example.should_answer and contexts)
                kwargs.update(reference=example.expected_answer, retrieved_contexts=contexts)
            else:
                eligible = answerable and bool(contexts)
                kwargs["response"] = answer
                if name == "faithfulness":
                    kwargs["retrieved_contexts"] = contexts
            if not eligible:
                results[name] = {"value": None, "status": "ineligible"}
                continue
            try:
                self.current_stage = f"ragas_{name}"
                # Each internal LLM call has its own timeout, pacing and bounded retry.
                # A whole-metric 120s timeout would incorrectly include deliberate waits.
                result = await metric.ascore(**kwargs)
                value = float(result.value)
                if not math.isfinite(value):
                    raise ValueError("Non-finite judge score")
                results[name] = {"value": value, "status": "ok"}
            except Exception as exc:
                # Avoid persisting SDK exception text, which may contain request credentials.
                results[name] = {"value": None, "status": "error", "error_type": type(exc).__name__,
                                 "diagnostic": diagnostic(exc, f"ragas_{name}")}
        return results

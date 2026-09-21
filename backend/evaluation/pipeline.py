"""Thin orchestration of existing production functions, with evaluation-only scope."""
import time
from uuid import uuid4


class EvaluationScope:
    def __init__(self, fixture_ids: list[str]):
        self.owner_id = f"policycue-eval-{uuid4().hex}"
        self.documents = {fixture: f"{self.owner_id}-{uuid4().hex}" for fixture in fixture_ids}
        self.attempted: set[str] = set()

    def guard(self, document_id: str, owner_id: str):
        if (owner_id != self.owner_id or not owner_id.startswith("policycue-eval-")
                or document_id not in self.documents.values()
                or not document_id.startswith(f"{owner_id}-")):
            raise ValueError("Refusing operation outside this evaluation run's scope")

    def cleanup(self, delete) -> list[dict]:
        results = []
        for document_id in sorted(self.attempted):
            self.guard(document_id, self.owner_id)
            try:
                delete(document_id, self.owner_id)
                results.append({"document_id": document_id, "status": "delete_requested"})
            except Exception as exc:
                results.append({"document_id": document_id, "status": "error",
                                "error_type": type(exc).__name__})
        return results


def wait_for_vectors(index, document_id: str, owner_id: str, chunks: list[str], timeout: float = 30):
    """Bounded visibility check before scoring; fail instead of scoring absent uploads."""
    deadline = time.monotonic() + timeout
    ids = [f"{document_id}-{i}" for i in range(len(chunks))]
    while True:
        visible = {}
        for start in range(0, len(ids), 100):
            visible.update(index.fetch(ids=ids[start:start + 100])["vectors"])
        if all(vector_id in visible and
               visible[vector_id]["metadata"].get("owner_id") == owner_id and
               visible[vector_id]["metadata"].get("document_id") == document_id and
               visible[vector_id]["metadata"].get("text") == chunks[i]
               for i, vector_id in enumerate(ids)):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Evaluation vectors not visible before deadline")
        time.sleep(1)


class PipelineFailure(RuntimeError):
    def __init__(self, stage, partial, error):
        super().__init__(f"Evaluation stage failed: {stage}")
        self.stage, self.partial, self.error = stage, partial, error


def run_question(example, document_id: str, owner_id: str, k: int, reliability=None) -> dict:
    from unittest.mock import patch
    from backend.app.analysis import answer_generator
    from backend.app.retrieval import embeddings, vector_store
    from backend.evaluation.reliability import Reliability, ReliabilityConfig, diagnostic

    policy = reliability or Reliability(ReliabilityConfig(0, 0, 0))
    result = {"stages": {}}

    def stage_call(operation, stage, kind=None):
        try:
            value = policy.call(operation, stage, kind)
            result["stages"][stage] = {"status": "ok"}
            return value
        except Exception as exc:
            result["stages"][stage] = {"status": "error", "diagnostic": diagnostic(exc, stage)}
            raise PipelineFailure(stage, result, exc) from exc

    # Temporarily substitute ONLY retry boundaries in this sequential CLI process.
    # Search/embedding algorithms, prompts, models and arguments remain production's.
    # The context managers restore every binding, including on failure.
    with patch.object(embeddings, "retry_external", lambda op: stage_call(op, "query_embedding")), patch.object(
            vector_store, "retry_external", lambda op: stage_call(op, "pinecone_retrieval")):
        try:
            chunks = vector_store.search_chunks(query=example.question, top_k=k,
                                                 document_id=document_id, owner_id=owner_id)
        except PipelineFailure:
            raise
        except Exception as exc:
            raise PipelineFailure("retrieval_mapping", result, exc) from exc
    for chunk in chunks:
        if chunk["document_id"] != document_id or chunk.get("owner_id") != owner_id:
            raise ValueError("Retrieved chunk outside evaluation scope")
    result.update(retrieved=chunks, document_found=bool(chunks))
    result["stages"]["retrieval"] = {"status": "ok"}
    with patch.object(answer_generator, "retry_external",
                      lambda op: stage_call(op, "generation", "generation")):
        try:
            answer = answer_generator.generate_answer(example.question, chunks)
        except PipelineFailure:
            raise
        except Exception as exc:
            raise PipelineFailure("generation", result, exc) from exc
    answerable = answer != answer_generator.NO_ANSWER_MESSAGE
    result.update(answer=answer, answerable=answerable, sources=chunks if answerable else [])
    result["stages"]["generation"] = {"status": "ok"}
    result["source_attributions"] = stage_call(
        lambda: vector_store.source_attributions(result["sources"]), "attribution")
    return result


def labeled_chunk_ids(chunks: list[dict], fixture_id: str) -> list[str]:
    return [f"{fixture_id}:{int(chunk['chunk_index'])}" for chunk in chunks]


def verify_cleanup(scope, index, chunks, timeout=30) -> dict:
    """Read only exact registered IDs, with bounded eventual-consistency polling."""
    from datetime import datetime, timezone
    from backend.evaluation.reliability import diagnostic

    ids = []
    for fixture, document in scope.documents.items():
        if document in scope.attempted:
            scope.guard(document, scope.owner_id)
            ids.extend(f'{document}-{i}' for i in range(len(chunks[fixture])))
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = []
            for start in range(0, len(ids), 100):
                remaining.extend(index.fetch(ids=ids[start:start + 100])['vectors'])
            if not remaining or time.monotonic() >= deadline:
                return {'status': 'absent' if not remaining else 'still_visible',
                        'checked_count': len(ids), 'remaining_vector_ids': remaining,
                        'checked_at': datetime.now(timezone.utc).isoformat()}
            time.sleep(2)
    except Exception as exc:
        return {'status': 'error', 'checked_count': len(ids),
                'diagnostic': diagnostic(exc, 'cleanup_verification')}

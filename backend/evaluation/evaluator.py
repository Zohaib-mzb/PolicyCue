"""Run from the repository root: python -m backend.evaluation.evaluator --help."""
import argparse
import asyncio
import hashlib
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from backend.evaluation.dataset import load_dataset, prepare_chunks
from backend.evaluation.pipeline import EvaluationScope, PipelineFailure, labeled_chunk_ids, run_question, wait_for_vectors, verify_cleanup
from backend.evaluation.ragas_metrics import NAMES, RagasJudge
from backend.evaluation.reliability import Reliability, ReliabilityConfig, diagnostic
from backend.evaluation.report import aggregate, summary, write_report, metric_coverage
from backend.evaluation.retrieval_metrics import abstention_metrics, citation_metrics, retrieval_metrics

ROOT = Path(__file__).parent


async def evaluate(
    dataset, k: int, judge=None, progress: Callable[[str], None] | None = None, reliability=None, checkpoint=None,
    resume_report=None,
) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    chunks = prepare_chunks(dataset)
    from backend.app.core.config import get_settings
    from backend.app.retrieval import vector_store

    settings = get_settings()
    scope = EvaluationScope(list(chunks))
    report = {
        "schema_version": 2, "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": dataset.model_dump(),
        "dataset_sha256": hashlib.sha256(dataset.model_dump_json().encode()).hexdigest(),
        "k": k, "precision_denominator": "k", "examples": len(dataset.examples),
        "answerable_examples": sum(ex.should_answer for ex in dataset.examples),
        "owner_id": scope.owner_id, "document_mapping": scope.documents,
        "configuration": {"generation_model": settings.gemini_model,
                          "judge_model": judge.model if judge else None,
                          "embedding_model": settings.pinecone_embedding_model,
                          "embedding_dimension": settings.pinecone_embedding_dimension,
                          "reliability": asdict(reliability.config) if reliability else None},
        "packages": {}, "rows": [], "status": "ok", "cleanup": [],
    }
    for package in ("ragas", "instructor", "google-genai", "pinecone", "langchain-text-splitters"):
        try:
            report["packages"][package] = version(package)
        except PackageNotFoundError:
            report["packages"][package] = None
    if resume_report is not None:
        expected_hash = hashlib.sha256(dataset.model_dump_json().encode()).hexdigest()
        if (resume_report.get("dataset_sha256") != expected_hash or resume_report.get("k") != k
                or set(resume_report.get("document_mapping", {})) != set(chunks)
                or [row.get("id") for row in resume_report.get("rows", [])]
                != [example.id for example in dataset.examples[:len(resume_report.get("rows", []))]]
                or len(resume_report.get("rows", [])) >= len(dataset.examples)):
            raise ValueError("Resume checkpoint does not match frozen dataset, k, fixtures, or row order")
        if any(row.get("status") != "ok" for row in resume_report["rows"]):
            raise ValueError("Resume requires completed prior rows")
        scope.owner_id = resume_report["owner_id"]
        scope.documents = resume_report["document_mapping"]
        for document_id in scope.documents.values():
            scope.guard(document_id, scope.owner_id)
        report = resume_report.copy()
        report["rows"] = list(resume_report["rows"])
        report["request_failures"] = list(resume_report.get("request_failures", []))
        report["cleanup"] = []
        report.pop("cleanup_verification", None)
        report["status"] = "ok"
        report.setdefault("execution_segments", [{"started_at": report["started_at"],
            "completed_rows": len(report["rows"]), "status": "interrupted"}])
        report["execution_segments"].append({"started_at": datetime.now(timezone.utc).isoformat(),
            "starting_row": len(report["rows"]), "status": "running"})
    try:
        for fixture, texts in chunks.items():
            document_id = scope.documents[fixture]
            scope.guard(document_id, scope.owner_id)
            scope.attempted.add(document_id)  # Includes partially failed upserts.
            vector_store.store_chunks(document_id, texts, owner_id=scope.owner_id,
                                      chunk_metadata=[{"source_type": "text", "title": fixture} for _ in texts])
            wait_for_vectors(vector_store.index, document_id, scope.owner_id, texts)
        for example in dataset.examples[len(report["rows"]):]:
            row = {"id": example.id, "status": "ok"}
            if reliability:
                reliability.example_id = example.id
            try:
                args = (example, scope.documents[example.document_id], scope.owner_id, k)
                actual = run_question(*args, reliability=reliability) if reliability else run_question(*args)
                row.update(actual)
            except PipelineFailure as exc:
                row.update(exc.partial, status="error", error_type=type(exc.error).__name__,
                           diagnostic=diagnostic(exc.error, exc.stage))
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__,
                           diagnostic=diagnostic(exc, "pipeline"))
            try:
                if "retrieved" in row:
                    ids = labeled_chunk_ids(row["retrieved"], example.document_id)
                    row["retrieved_chunk_ids"] = ids
                    row["retrieval"] = retrieval_metrics(ids, example.relevant_chunk_ids, k)
                if "answerable" in row:
                    row["abstention"] = abstention_metrics(example.should_answer, row["answerable"])
                if "sources" in row:
                    ids = labeled_chunk_ids(row["sources"], example.document_id)
                    row["cited_chunk_ids"] = ids
                    row["citations"] = citation_metrics(ids, example.relevant_chunk_ids, example.should_answer)
                if "answer" in row:
                    row["semantic"] = (await judge.score(example, row["answer"], row["answerable"],
                                           [item["text"] for item in row["retrieved"]])
                                       if judge else {name: {"value": None, "status": "disabled"} for name in NAMES})
                    row.setdefault("stages", {}).update({f"ragas_{name}": metric
                                                         for name, metric in row["semantic"].items()})
            except Exception as exc:
                row.update(status="error", error_type=type(exc).__name__,
                           diagnostic=diagnostic(exc, "metric_or_judge_orchestration"))
            if row["status"] == "error" or any(m["status"] == "error" for m in row.get("semantic", {}).values()):
                report["status"] = "partial"
            report["rows"].append(row)
            report["request_failures"] = (resume_report.get("request_failures", []) if resume_report else []) + (reliability.events if reliability else [])
            if checkpoint:
                checkpoint(report)
            if progress:
                progress(f"Completed {len(report['rows'])}/{len(dataset.examples)}: "
                         f"{example.id} ({row['status']})")
    except Exception as exc:
        report.update(status="error", error_type=type(exc).__name__, diagnostic=diagnostic(exc, "ingestion"))
    finally:
        report["cleanup"] = scope.cleanup(vector_store.delete_document_vectors)
        report["cleanup_verification"] = verify_cleanup(scope, vector_store.index, chunks)
        if report["cleanup_verification"]["status"] != "absent":
            report["status"] = "error"
        if any(item["status"] == "error" for item in report["cleanup"]):
            report["status"] = "error"
    report["summary"] = aggregate(report["rows"])
    report["metric_coverage"] = metric_coverage(dataset, report["rows"])
    report["not_run"] = len(dataset.examples) - len(report["rows"])
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    if resume_report is not None:
        report["execution_segments"][-1].update(completed_rows=len(report["rows"])-report["execution_segments"][-1]["starting_row"],
                                                  status="complete" if report["not_run"] == 0 else "interrupted")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Opt-in live PolicyCue evaluation (uses paid APIs unless --validate-only).")
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset.json")
    parser.add_argument("--k", type=int, default=5, help="Production default: 5")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-ragas", action="store_true", help="Still runs live Pinecone and Gemini generation")
    parser.add_argument("--request-delay", type=float, default=6.0)
    parser.add_argument("--judge-delay", type=float, default=6.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--judge-model", help="Defaults to configured GEMINI_MODEL")
    parser.add_argument("--validate-only", action="store_true", help="Validate labels and chunk manifests offline")
    parser.add_argument("--resume-checkpoint", type=Path, help="Resume a cleaned interrupted checkpoint without recomputing completed rows")
    args = parser.parse_args()
    if args.k < 1:
        parser.error("--k must be positive")
    dataset = load_dataset(args.dataset)
    if args.validate_only:
        print(f"Validated {dataset.name}/{dataset.version}: {len(dataset.examples)} examples")
        for fixture, texts in prepare_chunks(dataset).items():
            for i, text in enumerate(texts):
                print(f"{fixture}:{i}\n{text}\n")
        return 0
    resume_report = None
    if args.resume_checkpoint:
        resume_report = json.loads(args.resume_checkpoint.read_text(encoding="utf-8"))
        receipt_path = args.resume_checkpoint.with_name(args.resume_checkpoint.stem + "-cleanup.json")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (receipt.get("checkpoint") != args.resume_checkpoint.name
                or receipt.get("owner_id") != resume_report.get("owner_id")
                or receipt.get("verification", {}).get("status") != "absent"
                or len(receipt.get("delete_requests", [])) != len(resume_report.get("document_mapping", {}))
                or any(item.get("status") != "delete_requested" for item in receipt["delete_requests"])):
            parser.error("Resume requires successful exact-ID cleanup of the interrupted segment")
    try:
        reliability = Reliability(ReliabilityConfig(args.request_delay, args.judge_delay, args.max_retries))
    except ValueError as exc:
        parser.error(str(exc))
    judge = None
    if not args.skip_ragas:
        from backend.app.core.config import get_settings
        judge = RagasJudge(get_settings(), args.judge_model, reliability=reliability)
    checkpoint_path = args.output_dir / ("checkpoint-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + ".json")
    def checkpoint(report):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(checkpoint_path)

    async def run():
        try:
            report = await evaluate(
                dataset, args.k, judge,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                reliability=reliability, checkpoint=checkpoint,
                resume_report=resume_report,
            )
            path = write_report(report, args.output_dir)
            checkpoint(report)
            receipt = path.with_name(path.stem + "-cleanup.json")
            receipt.write_text(json.dumps(report["cleanup_verification"], indent=2) + "\n")
            print(summary(report))
            print(f"Report: {path}")
            return 0 if report["status"] == "ok" else 1
        finally:
            if judge:
                await judge.close()

    return asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(main())

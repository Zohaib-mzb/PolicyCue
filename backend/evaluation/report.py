"""JSON-safe aggregation, with explicit denominators and no invented scores."""
import json
from datetime import datetime, timezone
from pathlib import Path

from backend.evaluation.ragas_metrics import NAMES
from backend.evaluation.retrieval_metrics import mean


def aggregate(rows: list[dict]) -> dict:
    completed = [row for row in rows if row.get("status") == "ok"]
    retrieval_rows = [row for row in rows if "retrieval" in row]
    answer_rows = [row for row in rows if "abstention" in row]
    citation_rows = [row for row in rows if "citations" in row]
    retrieval = {name: mean(row["retrieval"][name] for row in retrieval_rows)
                 for name in ("recall", "precision", "reciprocal_rank")}
    retrieval["mrr"] = retrieval.pop("reciprocal_rank")
    abstention = {name: mean(row["abstention"][name] for row in answer_rows)
                  for name in ("abstention_accuracy", "answerable_accuracy", "unanswerable_accuracy")}
    abstention["confusion"] = {
        outcome: sum(row["abstention"]["outcome"] == outcome for row in answer_rows)
        for outcome in ("true_answer", "true_abstain", "false_answer", "false_abstain")}
    citations = {name: mean(row["citations"][name] for row in citation_rows)
                 for name in ("citation_correctness", "supporting_citation")}
    citations.update({name: sum(row["citations"][name] for row in citation_rows)
                      for name in ("correct_citation_count", "incorrect_citation_count", "missing_citation_count")})
    semantic = {}
    for name in NAMES:
        scores = [row["semantic"][name] for row in rows if name in row.get("semantic", {})]
        semantic[name] = mean(score["value"] for score in scores)
        semantic[name]["statuses"] = {status: sum(score["status"] == status for score in scores)
                                      for status in ("ok", "error", "ineligible", "disabled")}
    return {"pipeline_results": len(answer_rows), "retrieval_results": len(retrieval_rows),
            "completed": len(completed), "failed": len(rows) - len(completed),
            "retrieval": retrieval, "abstention": abstention, "citations": citations,
            "semantic": semantic}


def write_report(report: dict, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path = directory / f"evaluation-{stamp}.json"
    with path.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    return path


def summary(report: dict) -> str:
    data = report["summary"]
    lines = ["PolicyCue RAG Evaluation", f"Dataset: {report['dataset']['name']} / {report['dataset']['version']}",
             f"Status: {report['status']}; examples: {report['examples']}; answerable: {report['answerable_examples']}; "
             f"unanswerable: {report['examples'] - report['answerable_examples']}; k: {report['k']}"]
    if report["dataset"]["synthetic"]:
        lines.append("CONTROLLED SYNTHETIC BENCHMARK — not representative of all real-world policies")
    for group in ("retrieval", "semantic", "citations", "abstention"):
        lines.append(group.capitalize())
        for name, metric in data[group].items():
            if isinstance(metric, dict) and "value" in metric:
                value = "N/A" if metric["value"] is None else f"{metric['value']:.4f}"
                lines.append(f"  {name}: {value} (n={metric['count']})")
            else:
                lines.append(f"  {name}: {metric}")
    lines.append(f"Completed: {data['completed']}; failed: {data['failed']}; not run: {report['not_run']}")
    cleanup_errors = [item for item in report["cleanup"] if item["status"] == "error"]
    lines.append(f"Cleanup: {len(report['cleanup']) - len(cleanup_errors)} delete requests accepted; "
                 f"{len(cleanup_errors)} errors (see report for exact document/owner IDs)")
    return "\n".join(lines)


def metric_coverage(dataset, rows: list[dict]) -> dict:
    """Explicit known eligibility and missing measurements, without changing formulas."""
    by_id = {row['id']: row for row in rows}
    result = {}
    metrics = [('retrieval', name) for name in ('recall','precision','reciprocal_rank')]
    metrics += [('abstention', name) for name in ('abstention_accuracy','answerable_accuracy','unanswerable_accuracy')]
    metrics += [('citations', name) for name in ('citation_correctness','supporting_citation')]
    metrics += [('semantic', name) for name in NAMES]
    for group, name in metrics:
        counts = dict(eligible=0, successful=0, errors=0, unavailable=0, ineligible=0, unknown_eligibility=0)
        for example in dataset.examples:
            row = by_id.get(example.id, {})
            if group == 'retrieval' or name in {'answerable_accuracy','supporting_citation'}:
                eligible = example.should_answer
            elif name == 'unanswerable_accuracy':
                eligible = not example.should_answer
            elif name == 'abstention_accuracy':
                eligible = True
            elif name == 'citation_correctness':
                eligible = bool(row['sources']) if 'sources' in row else None
            elif name == 'context_precision':
                eligible = (example.should_answer and bool(row['retrieved'])) if 'retrieved' in row else None
            else:
                eligible = (row['answerable'] and bool(row['retrieved'])) if 'answerable' in row else None
            if eligible is None:
                counts['unknown_eligibility'] += 1
                continue
            if not eligible:
                counts['ineligible'] += 1
                continue
            counts['eligible'] += 1
            metric = row.get(group, {}).get(name)
            if group == 'semantic':
                status = metric.get('status') if metric else None
                if status == 'ok': counts['successful'] += 1
                elif status == 'error': counts['errors'] += 1
                else: counts['unavailable'] += 1
            elif metric is not None:
                counts['successful'] += 1
            else:
                counts['unavailable'] += 1
        result[f'{group}.{name}'] = counts
    return result

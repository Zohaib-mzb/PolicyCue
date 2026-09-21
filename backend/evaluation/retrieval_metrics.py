"""Deterministic metrics. Precision uses k, even when fewer results are returned."""
from collections.abc import Iterable


def mean(values: Iterable[float | bool | None]) -> dict:
    eligible = [float(value) for value in values if value is not None]
    return {"value": sum(eligible) / len(eligible) if eligible else None, "count": len(eligible)}


def retrieval_metrics(retrieved: list[str], relevant: list[str], k: int) -> dict:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    if not relevant:
        return dict(recall=None, precision=None, reciprocal_rank=None)
    truth = set(relevant)
    top = retrieved[:k]
    hits = len(set(top) & truth)
    rr = next((1 / rank for rank, chunk in enumerate(top, 1) if chunk in truth), 0.0)
    return dict(recall=hits / len(truth), precision=hits / k, reciprocal_rank=rr)


def abstention_metrics(expected: bool, actual: bool) -> dict:
    return {
        "abstention_accuracy": expected == actual,
        "answerable_accuracy": actual if expected else None,
        "unanswerable_accuracy": not actual if not expected else None,
        "outcome": {(True, True): "true_answer", (False, False): "true_abstain",
                    (False, True): "false_answer", (True, False): "false_abstain"}[expected, actual],
    }


def citation_metrics(cited: list[str], relevant: list[str], should_answer: bool) -> dict:
    citations, truth = set(cited), set(relevant)
    correct = citations & truth
    return {
        "citation_correctness": len(correct) / len(citations) if citations else None,
        "supporting_citation": bool(correct) if should_answer else None,
        "correct_citation_count": len(correct),
        "incorrect_citation_count": len(citations - truth),
        # Missing labeled evidence, including partially cited multi-chunk answers.
        "missing_citation_count": len(truth - citations),
    }

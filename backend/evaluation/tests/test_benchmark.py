"""Structural quality checks, independent of the benchmark's exact wording."""
from collections import Counter

import pytest

from backend.evaluation.dataset import load_dataset, prepare_chunks
from backend.evaluation.evaluator import ROOT


@pytest.fixture(scope="module")
def benchmark():
    return load_dataset(ROOT / "dataset.json")


def test_benchmark_size_and_balance(benchmark):
    assert benchmark.name and benchmark.version
    assert benchmark.synthetic is True
    assert "controlled" in benchmark.description.lower()
    assert 45 <= len(benchmark.examples) <= 60
    counts = Counter(example.should_answer for example in benchmark.examples)
    assert 35 <= counts[True] <= 45
    assert 10 <= counts[False] <= 15
    assert len({example.question.casefold() for example in benchmark.examples}) == len(benchmark.examples)


def test_documents_create_top_five_competition(benchmark):
    chunks = prepare_chunks(benchmark)  # Validates every ordered SHA-256 manifest.
    assert 5 <= len(chunks) <= 8
    assert all(len(texts) > 5 for texts in chunks.values())
    assert all(len(text) <= 1200 for texts in chunks.values() for text in texts)
    for doc in benchmark.documents:
        assert doc.text == (ROOT / "fixtures" / f"{doc.id}.txt").read_text(encoding="utf-8").strip()


def test_question_types_and_evidence(benchmark):
    categories = Counter(example.category for example in benchmark.examples)
    assert set(categories) >= {"direct_fact", "paraphrased", "qualifier_sensitive", "multi_sentence",
                               "multi_chunk", "semantic_confusion", "unanswerable", "negative_exception"}
    assert sum(len(example.relevant_chunk_ids) > 1 for example in benchmark.examples) >= 8
    assert any(len(example.relevant_chunk_ids) >= 3 for example in benchmark.examples)
    for document in benchmark.documents:
        examples = [example for example in benchmark.examples if example.document_id == document.id]
        assert any(example.should_answer for example in examples)
        assert any(not example.should_answer for example in examples)
    for example in benchmark.examples:
        assert example.notes and example.category
        if example.category == "multi_chunk":
            assert len(example.relevant_chunk_ids) > 1
        if not example.should_answer:
            assert example.expected_answer is None
            assert example.relevant_chunk_ids == []

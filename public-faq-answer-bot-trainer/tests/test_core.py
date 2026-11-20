import json

import pytest

from app import core, storage


def test_dataset_loads_and_is_balanced():
    faqs = core.load_faqs()
    assert len(faqs) >= 50
    counts = core.department_counts(faqs)
    assert len(counts) >= 8
    assert all(n >= 2 for n in counts.values())


def test_every_fact_is_in_its_answer():
    for faq in core.load_faqs():
        for fact in faq.facts:
            assert fact.lower() in faq.answer.lower()


def test_split_holds_out_last_paraphrase():
    faqs = core.load_faqs()
    train, held_out = core.split_examples(faqs)
    assert len(held_out) == len(faqs)
    train_questions = {e.question for e in train}
    assert not train_questions & {e.question for e in held_out}
    for faq in faqs:
        assert faq.question in train_questions


def test_prompt_and_target_format():
    prompt = core.format_prompt("  How do I   renew? ")
    assert prompt == "Riverton City Services FAQ\nQuestion: How do I renew?\nAnswer:"
    assert core.format_target("Online.") == " Online."


def _write(tmp_path, rows):
    path = tmp_path / "faqs.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


GOOD = {"id": "x-1", "department": "Test", "question": "Q one?", "paraphrases": ["Q uno?"],
        "answer": "It costs $5.", "facts": ["$5"]}


def test_fact_missing_from_answer_is_rejected(tmp_path):
    bad = dict(GOOD, facts=["$9"])
    with pytest.raises(ValueError, match="does not appear"):
        core.load_faqs(_write(tmp_path, [bad]))


def test_needs_a_paraphrase(tmp_path):
    with pytest.raises(ValueError, match="paraphrase"):
        core.load_faqs(_write(tmp_path, [dict(GOOD, paraphrases=[])]))


def test_duplicate_questions_rejected(tmp_path):
    other = dict(GOOD, id="x-2", paraphrases=["Something else?"])
    with pytest.raises(ValueError, match="appears in both"):
        core.load_faqs(_write(tmp_path, [GOOD, other]))


def test_bad_json_line_reports_line_number(tmp_path):
    path = tmp_path / "faqs.jsonl"
    path.write_text(json.dumps(GOOD) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        core.load_faqs(path)


def test_feedback_log_roundtrip(tmp_path):
    path = tmp_path / "fb.jsonl"
    storage.record_feedback("q", "a", 4, "fine", model="merged", path=path)
    rows = storage.read_feedback(path)
    assert rows[0]["rating"] == 4 and rows[0]["comment"] == "fine"
    with pytest.raises(ValueError):
        storage.record_feedback("q", "a", 9, path=path)

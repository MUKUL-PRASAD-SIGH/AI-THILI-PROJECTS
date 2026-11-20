import pytest

from app import core, evaluate, llm


@pytest.fixture(scope="module")
def faqs():
    return core.load_faqs()


def test_token_f1_and_rouge():
    assert evaluate.token_f1("The fee is $32.", "the fee is $32") == 1.0
    assert evaluate.token_f1("nothing alike", "the fee is $32") == 0.0
    assert evaluate.rouge_l("a b c d", "a c d") == pytest.approx(2 * (3 / 4) * 1 / (3 / 4 + 1))
    assert 0 < evaluate.token_f1("fee is $40", "the fee is $32") < 1


def test_fact_hits_ignore_case_and_punctuation():
    hits = evaluate.fact_hits("Renew ONLINE within 90 days; fee: $32", ["online", "90 days", "$32", "12 Elm Street"])
    assert hits == [True, True, True, False]


def test_unsupported_numbers_flags_invented_fees():
    ref = "The renewal fee is $32 and you have 90 days."
    assert evaluate.unsupported_numbers("The fee is $32.", ref) == set()
    assert evaluate.unsupported_numbers("The fee is $45, call 555-0199.", ref) == {"$45", "555-0199"}


def test_perfect_answer_scores_full(faqs):
    _, held = core.split_examples(faqs)
    for ex in held:
        s = evaluate.score_answer(ex.answer, ex, faqs)
        assert s["fact_recall"] == s["all_facts"] == s["relevant"] == s["exact"] == 1.0
        assert s["unsupported"] == 0.0


def test_wrong_faq_answer_is_not_relevant(faqs):
    _, held = core.split_examples(faqs)
    ex = next(e for e in held if e.faq_id == "dmv-03")
    other = next(f for f in faqs if f.id == "lib-02").answer
    s = evaluate.score_answer(other, ex, faqs)
    assert s["relevant"] == 0.0 and s["fact_recall"] < 1.0


def test_lookup_baseline_finds_its_own_training_question(faqs):
    train, _ = core.split_examples(faqs)
    lookup = evaluate.LookupBaseline(train)
    for ex in train:
        assert lookup.match(ex.question).faq_id == ex.faq_id


def test_summarize_and_report(faqs, tmp_path):
    _, held = core.split_examples(faqs)
    good = evaluate.evaluate_predictions("LoRA test", [e.answer for e in held], held, faqs)
    bad = evaluate.evaluate_predictions("Empty", ["" for _ in held], held, faqs)
    assert good["summary"]["fact_recall"] == 1.0 and bad["summary"]["fact_recall"] == 0.0
    text = evaluate.write_report({"splits": {"Held-out paraphrases": [bad, good]}}, tmp_path / "r.md")
    assert "| LoRA test | 100.0%" in text and "No misses" in text


def test_clean_answer_stops_at_next_question():
    assert llm.clean_answer(" Pay online.\nQuestion: What else?") == "Pay online."
    assert llm.clean_answer(" Pay online. Question: x") == "Pay online."

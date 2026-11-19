import io

import pytest

from app import storage
from app.cli import get_model, main, run_draft
from app.core import NGramModel, complete, current_context, suggest_next


@pytest.fixture(scope="module")
def model():
    return NGramModel().fit(storage.load_corpus())


@pytest.fixture
def model_file(tmp_path):
    path = tmp_path / "model.json"
    assert main(["train", "--model", str(path)]) == 0
    return str(path)


def test_prepare_and_train(tmp_path, capsys):
    corpus = tmp_path / "corpus.txt"
    assert main(["prepare", "--output", str(corpus)]) == 0
    assert main(["train", "--corpus", str(corpus), "--model", str(tmp_path / "m.json")]) == 0
    out = capsys.readouterr().out
    assert "anonymized sentences" in out and "perplexity" in out


def test_get_model_trains_when_missing(tmp_path):
    path = tmp_path / "missing.json"
    m = get_model(path)
    assert path.exists() and m.vocab_size > 100


def test_suggest_command_prints_top_k(model_file, capsys):
    assert main(["suggest", "Thank you for your", "-k", "4", "--model", model_file]) == 0
    out = capsys.readouterr().out
    assert out.startswith("[1] patience") and "[4]" in out


def test_generate_command(model_file, capsys):
    assert main(["generate", "Let me know if", "--model", model_file]) == 0
    assert capsys.readouterr().out.strip() == "Let me know if you need any further help."


def scripted(*entries):
    it = iter(entries)

    def fake_input(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return fake_input


def test_draft_accepts_suggestions_and_typed_text(model):
    out = io.StringIO()
    draft = run_draft(model, input_fn=scripted("Hi", "1", "", ":done"), output=out)
    assert draft == "Hi <NAME>,"  # the agent then replaces <NAME> with the customer's name
    assert "Final response" in out.getvalue()


def test_draft_select_from_top_k_and_undo(model):
    first = suggest_next(model, current_context("Thank you for"), k=5)[0][0]
    draft = run_draft(model, top_k=5, input_fn=scripted("Thank you for", "1", "2", ":undo", ":done"),
                      output=io.StringIO())
    assert draft == f"Thank you for {first}"
    second = suggest_next(model, current_context("Thank you for"), k=5)[1][0]
    draft = run_draft(model, top_k=5, input_fn=scripted("Thank you for", "2", ":done"), output=io.StringIO())
    assert draft == f"Thank you for {second}"


def test_draft_auto_complete_and_clear(model):
    draft = run_draft(model, input_fn=scripted("Let me check the", ":auto", ":done"), output=io.StringIO())
    assert draft == complete(model, "Let me check the")
    assert draft.startswith("Let me check the status of your") and draft.endswith(".")
    draft = run_draft(model, input_fn=scripted("hello", ":clear", ":done"), output=io.StringIO())
    assert draft == ""


def test_draft_rejects_out_of_range_choice(model):
    out = io.StringIO()
    draft = run_draft(model, top_k=3, input_fn=scripted("Thank you", "9", ":done"), output=out)
    assert draft == "Thank you" and "Pick a number" in out.getvalue()

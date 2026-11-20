import io

import pytest

from app.cli import format_predictions, get_predictor, main, parse_actions, run_live
from app.simulate import ACTIONS


@pytest.fixture(scope="module")
def markov():
    return get_predictor("markov", "unused")


def scripted(*entries):
    it = iter(entries)

    def fake_input(prompt):
        try:
            return next(it)
        except StopIteration:
            raise EOFError
    return fake_input


def test_parse_actions_accepts_aliases_and_separators():
    assert parse_actions("Inbound, bring up -> PnR; 3PT Made") == ["inbound", "bring_up", "pick_and_roll", "three_made"]
    assert parse_actions("") == []
    with pytest.raises(ValueError):
        parse_actions("moonwalk")


def test_predict_command_prints_ranked_confidences(capsys):
    assert main(["predict", "inbound, bring up, pick and roll", "--team", "Comets", "--model", "markov"]) == 0
    out = capsys.readouterr().out
    assert "1. drive" in out and "%" in out and "#" in out


def test_predict_rejects_unknown_team():
    with pytest.raises(SystemExit):
        main(["predict", "inbound", "--team", "Lakers", "--model", "markov"])


def test_format_predictions_orders_by_probability(markov):
    probs = markov.predict_proba([("Sharks", ["inbound"])])[0]
    lines = format_predictions(probs, k=3).splitlines()
    assert len(lines) == 3 and lines[0].split()[1] == ACTIONS[int(probs.argmax())]


def test_live_accepts_predictions_typed_actions_and_undo(markov):
    out = io.StringIO()
    history = run_live(markov, team="Comets", input_fn=scripted("inbound", "", "pick and roll", "1", "undo"),
                       output=out)
    assert history == ["inbound", "bring_up", "pick_and_roll"]
    assert "Comets: inbound -> bring_up" in out.getvalue()


def test_live_switches_team_and_reports_bad_input(markov):
    out = io.StringIO()
    history = run_live(markov, input_fn=scripted("inbound", "team titans", "moonwalk", "9", "post up"), output=out)
    assert history == ["post_up"]
    text = out.getvalue()
    assert "Titans:" in text and "unknown action" in text and "Pick a number" in text


def test_mine_command_writes_report(capsys, tmp_path, monkeypatch):
    from app import storage

    # write to a temp folder: the committed reports/patterns.md is the full top-10 report
    monkeypatch.setattr(storage, "REPORTS", tmp_path)
    assert main(["mine", "--top", "3"]) == 0
    assert "Team signatures" in capsys.readouterr().out
    assert (tmp_path / "patterns.md").exists()

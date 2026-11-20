import json

import pytest

from app.core import LABELS
from app.llm import Prediction
from app.storage import (load_corrections, load_saved, read_requests, save_correction,
                         save_predictions)


def pred(text="Is the hall free?", label="venue_inquiry", conf=0.9):
    scores = {l: (1 - conf) / (len(LABELS) - 1) for l in LABELS}
    scores[label] = conf
    return Prediction(text, label, conf, scores)


@pytest.mark.parametrize("name", ["out.jsonl", "out.csv"])
def test_save_predictions_appends_structured_rows(tmp_path, name):
    path = tmp_path / "sub" / name
    save_predictions(path, [pred()])
    save_predictions(path, [pred("Book a DJ", "entertainment_booking", 0.7)], source="batch")
    rows = load_saved(path)
    assert [r["category"] for r in rows] == ["venue_inquiry", "entertainment_booking"]
    assert float(rows[0]["confidence"]) == 0.9 and rows[1]["source"] == "batch"
    assert rows[0]["runner_up"] in LABELS and rows[0]["timestamp"]
    if name.endswith(".csv"):
        assert path.read_text(encoding="utf-8").count("timestamp,request") == 1  # one header


def test_read_requests_txt_and_csv(tmp_path):
    (tmp_path / "r.txt").write_text("# comment\nBook a DJ\n\n  Need a venue  \n", encoding="utf-8")
    assert read_requests(tmp_path / "r.txt") == ["Book a DJ", "Need a venue"]
    (tmp_path / "r.csv").write_text('id,request\n1,"Book a DJ, please"\n', encoding="utf-8")
    assert read_requests(tmp_path / "r.csv") == ["Book a DJ, please"]
    (tmp_path / "bad.csv").write_text("id,body\n1,x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_requests(tmp_path / "bad.csv")


def test_corrections_round_trip_latest_wins(tmp_path):
    path = tmp_path / "corrections.jsonl"
    save_correction("Need ice sculptures", "catering_request", "decor", path)
    save_correction("Need ice sculptures", "catering_request", "decor_request", path)
    save_correction("Can we start later?", "venue_inquiry", "schedule change", path)
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    assert rows[0]["corrected"] == "decor_request"
    ex = load_corrections(path)
    assert sorted(e.label for e in ex) == ["decor_request", "schedule_change"]


def test_correction_rejects_unknown_category(tmp_path):
    with pytest.raises(ValueError):
        save_correction("x", "venue_inquiry", "weather", tmp_path / "c.jsonl")

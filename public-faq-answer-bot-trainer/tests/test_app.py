"""The Streamlit app renders every tab without loading a model or raising."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


def test_app_renders(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))  # no trained models -> base only, nothing loaded
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Ask the bot", "FAQ dataset", "Train", "Evaluation",
                                          "Merge for deployment", "Feedback"]
    assert at.sidebar.radio[0].options == ["GPT-2, no fine-tuning"]


def test_picking_example_fills_question(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    radio = next(r for r in at.radio if r.key == "picked")
    radio.set_value(radio.options[0]).run()
    assert at.text_input(key="question").value == radio.options[0]

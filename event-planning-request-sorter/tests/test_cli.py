from app.cli import main
from app.storage import load_saved

TEXT = "note 7, please help with photos"


def run(tiny_trained, *argv, input_fn=input):
    out, _ = tiny_trained
    return main([argv[0], "--model-dir", str(out), "--device", "cpu", *argv[1:]],
                input_fn=input_fn)


def test_categories(capsys):
    assert main(["categories"]) == 0
    assert "venue_inquiry" in capsys.readouterr().out


def test_classify_one_request_and_save(tiny_trained, tmp_path, capsys):
    out = tmp_path / "results.csv"
    assert run(tiny_trained, "classify", TEXT, "--out", str(out)) == 0
    printed = capsys.readouterr().out
    assert "Category:   photo video" in printed and "confident" in printed
    rows = load_saved(out)
    assert rows[0]["request"] == TEXT and rows[0]["category"] == "photo_video"


def test_interactive_with_correction(tiny_trained, tmp_path, capsys):
    corrections = tmp_path / "c.jsonl"
    answers = iter([TEXT, "budget", "note 8, please help with band", "", ""])
    code = run(tiny_trained, "classify", "--corrections", str(corrections),
               input_fn=lambda _: next(answers))
    assert code == 0
    assert "Saved correction: photo video -> budget quote" in capsys.readouterr().out
    assert load_saved(corrections)[0]["corrected"] == "budget_quote"


def test_batch_file(tiny_trained, tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text("note 1, please help with venue\nnote 2, please help with shuttle\n",
                   encoding="utf-8")
    out = tmp_path / "sorted.jsonl"
    assert run(tiny_trained, "batch", str(src), "--out", str(out)) == 0
    assert "Saved 2 classified requests" in capsys.readouterr().out
    assert [r["category"] for r in load_saved(out)] == ["venue_inquiry", "guest_logistics"]


def test_correct_command(tiny_trained, tmp_path, capsys):
    path = tmp_path / "c.jsonl"
    assert run(tiny_trained, "correct", TEXT, "decor", "--corrections", str(path)) == 0
    assert load_saved(path)[0]["predicted"] == "photo_video"
    assert run(tiny_trained, "correct", TEXT, "nonsense") == 2


def test_missing_model_is_explained(tmp_path, capsys):
    assert main(["classify", "--model-dir", str(tmp_path), "hello"]) == 1
    assert "train" in capsys.readouterr().err

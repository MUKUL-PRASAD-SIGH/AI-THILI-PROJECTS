from app.cli import main


def test_describe_with_flags_saves_variants(tiny_model_dir, tmp_path, capsys):
    out = tmp_path / "out" / "descriptions.txt"
    code = main(["describe", "--model-dir", str(tiny_model_dir), "--device", "cpu",
                 "--name", "Cedar & Sage", "--notes", "cedarwood, sage", "--size", "8 oz",
                 "--length", "short", "--variants", "2", "--seed", "3", "--out", str(out)])
    assert code == 0
    printed = capsys.readouterr().out
    assert "Cedar & Sage (short)" in printed and "Saved" in printed
    saved = out.read_text(encoding="utf-8")
    assert "name: Cedar & Sage | notes: cedarwood, sage | size: 8 oz | length: short" in saved


def test_describe_interactive(tiny_model_dir, tmp_path, capsys):
    answers = iter(["Honey Glow", "honey, vanilla", "", "long", "2", str(tmp_path / "o.txt"), ""])
    code = main(["describe", "--model-dir", str(tiny_model_dir), "--device", "cpu", "--seed", "1"],
                input_fn=lambda _: next(answers))
    assert code == 0
    assert "Honey Glow (long)" in capsys.readouterr().out
    assert "size: 8 oz" in (tmp_path / "o.txt").read_text(encoding="utf-8")


def test_describe_without_model_explains_what_to_do(tmp_path, capsys):
    assert main(["describe", "--model-dir", str(tmp_path), "--name", "X"]) == 1
    assert "train" in capsys.readouterr().err


def test_tokenize_shows_pieces(tiny_model_dir, capsys):
    assert main(["tokenize", "--model-dir", str(tiny_model_dir), "warm vanilla candle"]) == 0
    assert "tokens for" in capsys.readouterr().out


def test_bad_variants_rejected(capsys):
    assert main(["describe", "--name", "X", "--variants", "0"]) == 2

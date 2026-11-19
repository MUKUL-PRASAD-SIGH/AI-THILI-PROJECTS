from app.cli import main


def test_hello(capsys):
    assert main(["hello"]) == 0
    assert "Hello" in capsys.readouterr().out

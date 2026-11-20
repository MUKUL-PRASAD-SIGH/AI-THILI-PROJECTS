import app


def test_package_imports():
    assert app.__doc__


def test_cli_parses_every_command():
    from app.cli import build_parser

    p = build_parser()
    assert p.parse_args(["train", "--r", "8"]).r == 8
    assert p.parse_args(["ask", "Hi?", "--model", "base"]).question == ["Hi?"]
    assert p.parse_args(["evaluate", "--skip-base"]).skip_base
    assert p.parse_args(["merge"]).command == "merge"
    assert p.parse_args(["sweep", "--only", "r=4"]).only == ["r=4"]

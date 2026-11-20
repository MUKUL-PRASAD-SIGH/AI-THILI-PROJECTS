"""Command line entry point (TODO)."""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    parser.add_argument("command", choices=["hello"])
    parser.parse_args(argv)
    print("Hello from the askThili starter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

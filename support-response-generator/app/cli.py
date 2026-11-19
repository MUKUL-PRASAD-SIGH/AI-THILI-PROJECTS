"""Command-line entry point."""
from __future__ import annotations

import argparse

from app import storage


def cmd_prepare(args) -> int:
    turns = storage.read_turns(args.input)
    speakers = () if args.all_speakers else ("agent",)
    lines = storage.build_corpus(turns, speakers)
    path = storage.write_corpus(lines, args.output)
    storage.write_anonymized_transcript(turns)
    print(f"Prepared {len(lines)} anonymized sentences from {len(turns)} turns -> {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-response-generator",
                                     description="Next-word suggestions for support agents.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the tool runs")

    p = sub.add_parser("prepare", help="clean + anonymize raw transcripts into a corpus")
    p.add_argument("--input", default=str(storage.RAW_PATH))
    p.add_argument("--output", default=str(storage.CORPUS_PATH))
    p.add_argument("--all-speakers", action="store_true", help="also train on customer turns")
    p.set_defaults(func=cmd_prepare)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from support-response-generator!")
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

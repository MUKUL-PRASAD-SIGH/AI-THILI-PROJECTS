"""Command-line entry point."""
from __future__ import annotations

import argparse

from app import storage
from app.core import NGramModel


def cmd_prepare(args) -> int:
    turns = storage.read_turns(args.input)
    speakers = () if args.all_speakers else ("agent",)
    lines = storage.build_corpus(turns, speakers)
    path = storage.write_corpus(lines, args.output)
    storage.write_anonymized_transcript(turns)
    print(f"Prepared {len(lines)} anonymized sentences from {len(turns)} turns -> {path}")
    return 0


def cmd_train(args) -> int:
    import random

    sentences = storage.load_corpus(args.corpus)
    rng = random.Random(0)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    cut = max(1, len(shuffled) // 10)
    held_out, train = shuffled[:cut], shuffled[cut:]
    heldout_model = NGramModel(n=args.n, k=args.k).fit(train)
    model = NGramModel(n=args.n, k=args.k).fit(sentences)
    path = storage.save_model(model, args.model)
    print(f"Trained {args.n}-gram model (add-k, k={args.k}) on {len(sentences)} sentences, "
          f"vocab={model.vocab_size} -> {path}")
    print(f"Held-out perplexity (10% split): {heldout_model.perplexity(held_out):.2f}")
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

    p = sub.add_parser("train", help="train the n-gram model on the prepared corpus")
    p.add_argument("--corpus", default=str(storage.CORPUS_PATH))
    p.add_argument("--model", default=str(storage.MODEL_PATH))
    p.add_argument("-n", type=int, default=3, help="n-gram order (default 3)")
    p.add_argument("-k", type=float, default=0.01, help="add-k smoothing constant (default 0.01)")
    p.set_defaults(func=cmd_train)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from support-response-generator!")
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Milestone 4: command-line interface.

    python -m app.cli categories
    python -m app.cli train [--epochs 10] [--with-corrections]
    python -m app.cli classify "Is the rooftop free on March 14th?" [--save] [--out file.csv]
    python -m app.cli classify                       # interactive: type requests one by one
    python -m app.cli batch requests.txt --out sorted.csv
    python -m app.cli correct "the request text" catering_request
"""
from __future__ import annotations

import argparse
import sys

from app.core import CATEGORIES, LABELS, pretty, resolve_label


def _bar(p: float, width: int = 20) -> str:
    n = round(p * width)
    return "#" * n + "." * (width - n)


def show(pred, top: int = 3, threshold: float = 0.5) -> None:
    print(f"\n  Request:    {pred.text}")
    print(f"  Category:   {pretty(pred.label)}  ({pred.confidence:.0%} confident)")
    if pred.confidence < threshold:
        print("  Note:       low confidence - worth a human look")
    if top > 1:
        for label, p in pred.top(top):
            print(f"    {pretty(label):<22} {_bar(p)} {p:6.1%}")


def _load_sorter(args):
    from app.llm import Sorter
    try:
        return Sorter(args.model_dir, args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return None


def _ask_correction(pred, input_fn, corrections_path) -> None:
    from app.storage import save_correction

    answer = input_fn("  Correct? [Enter = yes, or type the right category]: ").strip()
    if not answer or answer.lower() in ("y", "yes"):
        return
    try:
        label = resolve_label(answer)
    except ValueError as e:
        print(f"  {e}")
        return
    if label != pred.label:
        path = save_correction(pred.text, pred.label, label, corrections_path)
        print(f"  Saved correction: {pretty(pred.label)} -> {pretty(label)} ({path})")


def cmd_classify(args, input_fn=input) -> int:
    from app.storage import DEFAULT_CORRECTIONS, DEFAULT_RESULTS, save_predictions

    sorter = _load_sorter(args)
    if sorter is None:
        return 1
    out = args.out or (DEFAULT_RESULTS if args.save else None)
    corrections = args.corrections or DEFAULT_CORRECTIONS

    def handle(text: str, interactive: bool) -> None:
        pred = sorter.classify(text)
        show(pred, args.top, args.threshold)
        if interactive and args.review:
            _ask_correction(pred, input_fn, corrections)
        if out:
            save_predictions(out, [pred])
            print(f"  Saved to {out}")

    if args.text:
        handle(" ".join(args.text), interactive=False)
        return 0
    print("Type a client request and press Enter (empty line to quit).")
    while True:
        try:
            text = input_fn("\n> ").strip()
        except EOFError:
            return 0
        if not text:
            return 0
        handle(text, interactive=True)


def cmd_batch(args) -> int:
    from collections import Counter

    from app.storage import read_requests, save_predictions

    sorter = _load_sorter(args)
    if sorter is None:
        return 1
    try:
        texts = read_requests(args.file)
    except (OSError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    if not texts:
        print(f"No requests found in {args.file}", file=sys.stderr)
        return 1
    preds = sorter.classify_many(texts)
    width = min(60, max(len(t) for t in texts))
    for p in preds:
        flag = "  <- check" if p.confidence < args.threshold else ""
        text = p.text if len(p.text) <= width else p.text[: width - 3] + "..."
        print(f"{text:<{width}}  {pretty(p.label):<22} {p.confidence:6.1%}{flag}")
    counts = Counter(p.label for p in preds)
    print("\n" + ", ".join(f"{pretty(l)}: {n}" for l, n in counts.most_common()))
    if args.out:
        save_predictions(args.out, preds, source=f"batch:{args.file}")
        print(f"Saved {len(preds)} classified requests to {args.out}")
    return 0


def cmd_correct(args) -> int:
    from app.storage import DEFAULT_CORRECTIONS, save_correction

    try:
        label = resolve_label(args.category)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    predicted = ""
    sorter = _load_sorter(args) if not args.no_model else None
    if sorter is not None:
        predicted = sorter.classify(args.text).label
    path = save_correction(args.text, predicted, label, args.corrections or DEFAULT_CORRECTIONS)
    print(f"Saved correction ({pretty(predicted) or 'unknown'} -> {pretty(label)}) to {path}")
    return 0


def cmd_train(args) -> int:
    from app.storage import DEFAULT_CORRECTIONS, load_corrections
    from app.train import train

    extra = load_corrections(args.corrections or DEFAULT_CORRECTIONS) if args.with_corrections else None
    if args.with_corrections:
        print(f"adding {len(extra)} corrected requests to the training data")
    train(base=args.base_model, out_dir=args.model_dir, epochs=args.epochs, lr=args.lr,
          batch_size=args.batch_size, device=args.device, extra_examples=extra)
    return 0


def cmd_categories(_args) -> int:
    for label in LABELS:
        print(f"  {label:<22} {CATEGORIES[label]}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="event-planning-request-sorter",
        description="Sort client requests into categories with a fine-tuned GPT-2.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the tool runs")
    sub.add_parser("categories", help="list the request categories")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-dir", help="fine-tuned model folder (default: models/sorter)")
    common.add_argument("--device", help="cpu or cuda")
    common.add_argument("--corrections", help="corrections file (default: output/corrections.jsonl)")

    p = sub.add_parser("train", parents=[common], help="fine-tune GPT-2 on data/requests.csv")
    p.add_argument("--base-model", help="default: openai-community/gpt2 (or BASE_MODEL)")
    p.add_argument("--epochs", type=float, default=10)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--with-corrections", action="store_true",
                   help="also train on corrections you saved")

    p = sub.add_parser("classify", parents=[common],
                       help="classify one request (or type several interactively)")
    p.add_argument("text", nargs="*", help="the request; omit for interactive mode")
    p.add_argument("--save", action="store_true", help="append to output/classified.jsonl")
    p.add_argument("--out", help="append to this file (.jsonl or .csv)")
    p.add_argument("--top", type=int, default=3, help="show the top N categories")
    p.add_argument("--threshold", type=float, default=0.5, help="flag below this confidence")
    p.add_argument("--no-review", dest="review", action="store_false",
                   help="interactive mode: don't ask to confirm each category")

    p = sub.add_parser("batch", parents=[common], help="classify every request in a file")
    p.add_argument("file", help=".txt (one per line) or .csv with a 'text' column")
    p.add_argument("--out", help="save results here (.csv or .jsonl)")
    p.add_argument("--threshold", type=float, default=0.5)

    p = sub.add_parser("correct", parents=[common], help="record the right category for a request")
    p.add_argument("text")
    p.add_argument("category")
    p.add_argument("--no-model", action="store_true", help="don't run the model first")
    return parser


def main(argv=None, input_fn=input):
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from event-planning-request-sorter!")
        return 0
    handlers = {"categories": cmd_categories, "train": cmd_train, "batch": cmd_batch,
                "correct": cmd_correct}
    if args.command == "classify":
        return cmd_classify(args, input_fn)
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())

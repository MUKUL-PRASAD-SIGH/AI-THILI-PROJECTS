"""Milestone 6: command-line interface.

    python -m app.cli hello
    python -m app.cli build-corpus
    python -m app.cli train [--steps 1500]
    python -m app.cli describe --name "Midnight Fig" --notes "black fig, cedarwood" \
        --size "10 oz" --length long --variants 3 --out descriptions.txt
    python -m app.cli describe            # no --name: asks you step by step
    python -m app.cli tokenize "Hand-poured soy candle"
"""
from __future__ import annotations

import argparse
import sys

from app.core import LENGTHS, Product, parse_notes


def _ask(prompt: str, default: str = "", input_fn=input) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input_fn(f"{prompt}{suffix}: ").strip()
    return answer or default


def _print_results(product: Product, results: list[str]) -> None:
    print(f"\n{product.name} ({product.length})")
    print("-" * min(60, len(product.name) + len(product.length) + 3))
    for i, text in enumerate(results, 1):
        print(f"{i}. {text}\n" if len(results) > 1 else f"{text}\n")


def cmd_describe(args, input_fn=input) -> int:
    from app.llm import Describer
    from app.storage import save_descriptions

    try:
        describer = Describer(args.model_dir, args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    interactive = not args.name
    while True:
        if interactive:
            print("\nDescribe a product (leave the name empty to quit).")
            name = _ask("Product name", input_fn=input_fn)
            if not name:
                return 0
            notes = _ask("Scent notes, comma separated", input_fn=input_fn)
            size = _ask("Size", "8 oz", input_fn=input_fn)
            length = _ask("Length (short/medium/long)", args.length, input_fn=input_fn).lower()
            if length not in LENGTHS:
                print(f"Length must be one of {', '.join(LENGTHS)}; using {args.length}.")
                length = args.length
            variants = _ask("How many variants", str(args.variants), input_fn=input_fn)
            variants = int(variants) if variants.isdigit() and int(variants) > 0 else args.variants
        else:
            name, notes, size, length, variants = (args.name, args.notes or "", args.size,
                                                   args.length, args.variants)

        product = Product(name=name, notes=parse_notes(notes), size=size, length=length)
        results = describer.describe(product, n=variants, temperature=args.temperature,
                                     top_k=args.top_k, seed=args.seed,
                                     max_new_tokens=args.max_tokens)
        _print_results(product, results)

        out = args.out
        if interactive and not out:
            out = _ask("Save to file? (path, or Enter to skip)", input_fn=input_fn)
        if out:
            path = save_descriptions(out, product.header().split(">", 1)[1], results)
            print(f"Saved {len(results)} description(s) to {path}")
        if not interactive:
            return 0


def cmd_train(args) -> int:
    from app.storage import model_dir
    from app.train import TrainConfig, train

    cfg = TrainConfig()
    for field in ("steps", "batch_size", "block_size", "n_embd", "n_layer", "n_head",
                  "vocab_size", "lr"):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, "max_steps" if field == "steps" else field, value)
    cfg.eval_every = max(1, min(cfg.eval_every, cfg.max_steps))
    result = train(cfg, out_dir=model_dir(args.model_dir), device=args.device)
    print(f"done: best validation loss {result['best_val_loss']} "
          f"({result['params']:,} parameters, {result['seconds']}s)")
    return 0


def cmd_build_corpus(args) -> int:
    from app.corpus import CORPUS_FILE, build_corpus

    products = build_corpus(n_generated=args.n)
    print(f"wrote {len(products)} product records to {CORPUS_FILE}")
    return 0


def cmd_tokenize(args) -> int:
    from app.storage import load_tokenizer, model_dir

    try:
        tok = load_tokenizer(model_dir(args.model_dir))
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    pieces = tok.token_strings(args.text)
    print(" | ".join(repr(p)[1:-1] for p in pieces))
    print(f"{len(pieces)} tokens for {len(args.text.encode())} bytes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-product-describer",
        description="Write product descriptions with a GPT trained from scratch on your shop's texts.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the tool runs")

    p = sub.add_parser("build-corpus", help="clean the seed listings and build data/corpus.txt")
    p.add_argument("--n", type=int, default=2400, help="number of catalog products to generate")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model-dir", help="where the model is stored (default: models/)")
    common.add_argument("--device", help="cpu or cuda")

    p = sub.add_parser("train", parents=[common], help="train the tokenizer and GPT")
    p.add_argument("--steps", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--block-size", type=int)
    p.add_argument("--n-embd", type=int)
    p.add_argument("--n-layer", type=int)
    p.add_argument("--n-head", type=int)
    p.add_argument("--vocab-size", type=int)
    p.add_argument("--lr", type=float)

    p = sub.add_parser("describe", parents=[common],
                       help="generate descriptions (interactive if --name is omitted)")
    p.add_argument("--name", help="product name, e.g. 'Midnight Fig'")
    p.add_argument("--notes", help="comma-separated notes, e.g. 'black fig, cedarwood'")
    p.add_argument("--size", default="8 oz")
    p.add_argument("--length", choices=LENGTHS, default="medium",
                   help="desired description length")
    p.add_argument("--variants", type=int, default=1, help="how many different descriptions")
    p.add_argument("--out", help="append the descriptions to this text file")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--max-tokens", type=int, help="hard cap on generated tokens")
    p.add_argument("--seed", type=int, help="make the output repeatable")

    p = sub.add_parser("tokenize", parents=[common], help="show how the BPE tokenizer splits text")
    p.add_argument("text")
    return parser


def main(argv=None, input_fn=input):
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from ai-product-describer!")
        return 0
    if args.command == "build-corpus":
        return cmd_build_corpus(args)
    if args.command == "train":
        return cmd_train(args)
    if args.command == "describe":
        if args.variants < 1:
            print("--variants must be at least 1", file=sys.stderr)
            return 2
        return cmd_describe(args, input_fn)
    if args.command == "tokenize":
        return cmd_tokenize(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

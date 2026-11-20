"""Command line.

    python -m app.cli build-corpus               # data/corpus.txt from seeds + generator
    python -m app.cli train                      # small GPT -> models/gpt_fp32.pt
    python -m app.cli talk GUARD "Can I enter the town?"
    python -m app.cli talk                       # interactive: pick an NPC and chat
    python -m app.cli talk BARD "Play me a song." --model int4
    python -m app.cli quantize                   # dynamic / static / int8..int2 files and sizes
    python -m app.cli simulate                   # simulated speed on target devices
    python -m app.cli report                     # everything -> reports/quantization.md
"""
from __future__ import annotations

import argparse
import json

from app import core, corpus, storage


def cmd_hello(args) -> int:
    print("Hello from the Quantized Game Dialogue Simulator.")
    return 0


def cmd_build_corpus(args) -> int:
    stats = corpus.build_corpus(args.n, args.seed)
    print(json.dumps(stats, indent=2))
    print(f"wrote {storage.display(corpus.CORPUS_FILE)}")
    return 0


def cmd_train(args) -> int:
    from app import llm, train
    from app.model import GPTConfig

    llm.set_threads()
    mcfg = GPTConfig(n_layer=args.layers, n_head=args.heads, n_embd=args.embd, block_size=args.context)
    tcfg = train.TrainConfig(steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    _, history = train.train(mcfg, tcfg, out_dir=storage.model_dir())
    storage.write_json(storage.REPORTS / "train_history.json", history)
    return 0


def _load_variant(name: str):
    from app import llm

    if name == "fp32":
        model, tok, _ = llm.load_checkpoint()
        return model, tok
    from app import quantize

    return quantize.load_variant(name)


def cmd_talk(args) -> int:
    from app import llm

    llm.set_threads()
    model, tok = _load_variant(args.model)
    opts = dict(temperature=args.temperature, top_k=args.top_k, seed=args.seed)
    if args.role and args.line:
        for i in range(args.n):
            seed = None if args.seed is None else args.seed + i
            print(f"{args.role.upper()}: {llm.npc_reply(model, tok, args.role, args.line, **{**opts, 'seed': seed})}")
        return 0
    role = (args.role or "").upper()
    while role not in core.ROLES:
        role = input(f"Talk to which NPC? ({', '.join(core.ROLES)}): ").strip().upper()
    print(f"You are talking to the {role.lower()}. Empty line to leave.")
    while True:
        line = input("PLAYER: ").strip()
        if not line:
            return 0
        try:
            print(f"{role}: {llm.npc_reply(model, tok, role, line, **opts)}")
        except ValueError as exc:
            print(f"(can't say that: {exc})")


def cmd_quantize(args) -> int:
    from app import llm, quantize, train

    model, tok, _ = llm.load_checkpoint()
    _, train_ids, _ = train.load_data()
    variants = quantize.build_all(model, tok, train_ids)
    fp32 = variants["fp32"]["bytes"]
    print(f"\n{'variant':>8}  {'size':>10}  smaller by")
    for name, v in variants.items():
        print(f"{name:>8}  {v['bytes'] / 1e6:7.3f} MB  {fp32 / v['bytes']:5.2f}x")
    return 0


def cmd_simulate(args) -> int:
    from app import llm, report, simulate

    model, _, _ = llm.load_checkpoint()
    rows = simulate.simulate_devices(model, report.BITS, reply_tokens=args.reply_chars)
    for dev in simulate.DEVICES:
        print(f"\n{dev.name}: {dev.bandwidth_gbs} GB/s, {dev.fp32_gflops} GFLOP/s fp32, "
              f"{dev.int8_gops} GOP/s int8, {dev.budget_mb} MB budget")
        for r in [r for r in rows if r["device"] == dev.name]:
            fit = "fits" if r["fits"] else "DOES NOT FIT"
            print(f"  {r['variant']:>8}  {r['model_mb']:6.3f} MB  {fit:<12}  {r['ms_per_token']:8.4f} ms/token  "
                  f"{r['reply_ms']:8.1f} ms per {args.reply_chars}-char reply  ({r['bound']}-bound)")
    return 0


def cmd_report(args) -> int:
    from app import report

    report.run(samples_per_prompt=args.samples)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Quantized Game Dialogue Simulator")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("hello").set_defaults(func=cmd_hello)

    b = sub.add_parser("build-corpus", help="rebuild data/corpus.txt")
    b.add_argument("--n", type=int, default=6000, help="templated exchanges to draw before de-duplication")
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(func=cmd_build_corpus)

    t = sub.add_parser("train", help="train the dialogue GPT")
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=4)
    t.add_argument("--embd", type=int, default=128)
    t.add_argument("--context", type=int, default=128)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=cmd_train)

    k = sub.add_parser("talk", help="generate NPC replies")
    k.add_argument("role", nargs="?", help=", ".join(core.ROLES))
    k.add_argument("line", nargs="?", help="what the player says")
    k.add_argument("--model", default="fp32", help="fp32, dynamic, static, or int8/int4/... (see quantize)")
    k.add_argument("--n", type=int, default=1, help="number of replies to sample")
    k.add_argument("--temperature", type=float, default=0.8)
    k.add_argument("--top-k", type=int, default=20)
    k.add_argument("--seed", type=int, default=0)
    k.set_defaults(func=cmd_talk)

    q = sub.add_parser("quantize", help="build dynamic, static and int8..int2 models and compare file sizes")
    q.set_defaults(func=cmd_quantize)

    s = sub.add_parser("simulate", help="simulated speed and memory fit on target devices")
    s.add_argument("--reply-chars", type=int, default=60)
    s.set_defaults(func=cmd_simulate)

    r = sub.add_parser("report", help="full comparison: size, quality, measured + simulated speed -> reports/")
    r.add_argument("--samples", type=int, default=2, help="sampled replies per prompt, besides greedy")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

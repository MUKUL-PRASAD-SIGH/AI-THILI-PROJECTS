"""Command line: train the adapter, ask questions, evaluate, merge.

    python -m app.cli train            # LoRA fine-tune GPT-2 -> models/adapter
    python -m app.cli ask "How do I renew my driver's licence?" --model adapter
"""
from __future__ import annotations

import argparse
import sys

from app import core, lora, storage


def cmd_train(args) -> int:
    from app import llm, train

    faqs = core.load_faqs(args.data)
    train_ex, held_out = core.split_examples(faqs)
    print(f"{len(faqs)} FAQs: {len(train_ex)} training questions, {len(held_out)} held-out paraphrases")
    cfg = train.TrainConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
        lora=lora.LoRAConfig(r=args.r, alpha=args.alpha, dropout=args.dropout, target_modules=args.targets),
    )
    model, tok = llm.load_base(device=args.device)
    out = args.out or llm.adapter_dir()
    history = train.train_lora(model, tok, train_ex, held_out, cfg, device=str(model.device), out_dir=out,
                               extra_meta={"base_model": llm.base_model_name()})
    storage.write_json(f"{out}/history.json", history)
    return 0


def cmd_ask(args) -> int:
    from app import llm

    model, tok, kind = llm.load_model(args.model, device=args.device)
    questions = args.question or [line.strip() for line in sys.stdin if line.strip()]
    for q, a in zip(questions, llm.generate_answers(model, tok, questions, temperature=args.temperature)):
        print(f"[{kind}] Q: {q}\n       A: {a}\n")
    return 0


REPORTS = storage.PROJECT_ROOT / "reports"


def run_evaluation(model_kind: str = "auto", include_base: bool = True, log=print) -> dict:
    """Score base GPT-2, the lookup baseline and the LoRA model on held-out and seen questions."""
    from app import evaluate, llm

    faqs = core.load_faqs()
    train_ex, held_out = core.split_examples(faqs)
    seen = core.seen_examples(faqs)
    lookup = evaluate.LookupBaseline(train_ex)
    splits = {"Held-out paraphrases": held_out, "Seen questions": seen}
    results = {"splits": {name: [] for name in splits}}

    if include_base:
        log("answering with the pre-trained model (no adapter)...")
        base, tok = llm.load_base()
        for name, ex in splits.items():
            results["splits"][name].append(evaluate.evaluate_model("GPT-2, no fine-tuning", base, tok, ex, faqs))
        del base
    for name, ex in splits.items():
        preds = [lookup.answer(e.question) for e in ex]
        results["splits"][name].append(evaluate.evaluate_predictions("Word-overlap lookup", preds, ex, faqs))

    model, tok, kind = llm.load_model(model_kind)
    if kind == "base":
        raise SystemExit("no trained adapter found: run `python -m app.cli train` first")
    log(f"answering with the LoRA model ({kind})...")
    for name, ex in splits.items():
        results["splits"][name].append(evaluate.evaluate_model(f"LoRA GPT-2 ({kind})", model, tok, ex, faqs))

    cfg = lora.read_adapter_config(llm.adapter_dir())
    results["meta"] = {"base_model": cfg.get("base_model", llm.base_model_name()), "r": cfg["lora"]["r"],
                       "alpha": cfg["lora"]["alpha"], "targets": cfg["lora"]["target_modules"],
                       "n_faqs": len(faqs), "n_departments": len(core.department_counts(faqs)),
                       "model_evaluated": kind}
    storage.write_json(storage.model_dir() / "eval.json", results)
    storage.write_json(REPORTS / "eval_summary.json", {
        "meta": results["meta"],
        "splits": {s: [{"system": r["system"], **r["summary"]} for r in rs] for s, rs in results["splits"].items()},
        "held_out_rows": results["splits"]["Held-out paraphrases"][-1]["rows"],
    })
    evaluate.write_report(results, REPORTS / "evaluation.md")
    return results


def cmd_evaluate(args) -> int:
    results = run_evaluation(args.model, include_base=not args.skip_base)
    for split, systems in results["splits"].items():
        print(f"\n{split}")
        for r in systems:
            s = r["summary"]
            print(f"  {r['system']:<26} fact recall {s['fact_recall']:.1%}  all facts {s['all_facts']:.1%}  "
                  f"relevant {s['relevant']:.1%}  unsupported {s['unsupported']:.1%}  ROUGE-L {s['rouge_l']:.3f}")
    print(f"\nreport written to {REPORTS / 'evaluation.md'}")
    return 0


def cmd_merge(args) -> int:
    from app import merge

    report = merge.merge_and_save(args.adapter, args.out)
    storage.write_json(REPORTS / "merge_report.json", report)
    return 0


def cmd_sweep(args) -> int:
    """Stretch goal: compare LoRA ranks / alpha / target layers on held-out accuracy and cost."""
    from app import evaluate, llm, train

    faqs = core.load_faqs()
    train_ex, held_out = core.split_examples(faqs)
    attn = ["attn.c_attn", "attn.c_proj"]
    every = lora.LoRAConfig().target_modules
    grid = [
        ("r=1, attention only", lora.LoRAConfig(r=1, alpha=2, target_modules=attn)),
        ("r=16, attention only", lora.LoRAConfig(r=16, alpha=32, target_modules=attn)),
        ("r=4, all layers", lora.LoRAConfig(r=4, alpha=8, target_modules=every)),
        ("r=16, all layers", lora.LoRAConfig(r=16, alpha=32, target_modules=every)),
        ("r=16, all layers, alpha=16", lora.LoRAConfig(r=16, alpha=16, target_modules=every)),
    ]
    if args.only:
        grid = [g for g in grid if any(o in g[0] for o in args.only)]
    rows = []
    for label, lcfg in grid:
        print(f"\n=== {label}")
        model, tok = llm.load_base(device=args.device)
        cfg = train.TrainConfig(epochs=args.epochs, lr=args.lr, lora=lcfg)
        hist = train.train_lora(model, tok, train_ex, held_out, cfg, device=str(model.device))
        res = evaluate.evaluate_model(label, model, tok, held_out, faqs)
        rows.append({"config": label, "r": lcfg.r, "alpha": lcfg.alpha, "targets": lcfg.target_modules,
                     "trainable_params": hist["trainable_params"], "train_seconds": hist["train_seconds"],
                     "best_val_loss": hist["best_val_loss"], "best_epoch": hist["best_epoch"], **res["summary"]})
        storage.write_json(REPORTS / "lora_sweep.json", rows)
    lines = ["# LoRA configuration comparison", "",
             f"GPT-2, {args.epochs} epochs, lr {args.lr}, same data and seed. Scores are on the held-out paraphrases.",
             "", "| Config | Trainable params | Train time | Best held-out loss | Fact recall | All facts "
             "| Relevant | Unsupported numbers |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['config']} | {r['trainable_params']:,} | {r['train_seconds']:.0f}s | "
                     f"{r['best_val_loss']:.3f} | {r['fact_recall']:.1%} | {r['all_facts']:.1%} | "
                     f"{r['relevant']:.1%} | {r['unsupported']:.1%} |")
    (REPORTS / "lora_sweep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Public FAQ Answer Bot Trainer")
    p.add_argument("--device", default=None, help="cpu or cuda")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="LoRA fine-tune the base model on the FAQ dataset")
    t.add_argument("--data", default=str(core.DEFAULT_DATA))
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=8)
    t.add_argument("--lr", type=float, default=2e-3)
    t.add_argument("--r", type=int, default=16)
    t.add_argument("--alpha", type=float, default=32.0)
    t.add_argument("--dropout", type=float, default=0.05)
    t.add_argument("--targets", nargs="+", default=lora.LoRAConfig().target_modules)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None, help="adapter folder (default models/adapter)")
    t.set_defaults(func=cmd_train)

    a = sub.add_parser("ask", help="answer questions (arguments, or one per line on stdin)")
    a.add_argument("question", nargs="*")
    a.add_argument("--model", default="auto", choices=["auto", "merged", "adapter", "base"])
    a.add_argument("--temperature", type=float, default=0.0)
    a.set_defaults(func=cmd_ask)

    e = sub.add_parser("evaluate", help="score answers against the approved ones; writes reports/evaluation.md")
    e.add_argument("--model", default="auto", choices=["auto", "merged", "adapter"])
    e.add_argument("--skip-base", action="store_true", help="skip the slow pre-trained baseline")
    e.set_defaults(func=cmd_evaluate)

    m = sub.add_parser("merge", help="fold the adapter into the weights -> models/merged")
    m.add_argument("--adapter", default=None)
    m.add_argument("--out", default=None)
    m.set_defaults(func=cmd_merge)

    s = sub.add_parser("sweep", help="compare LoRA ranks, alpha and target layers")
    s.add_argument("--epochs", type=int, default=8, help="kept short: every config trains from scratch")
    s.add_argument("--only", nargs="*", help="run only configs whose label contains one of these")
    s.add_argument("--lr", type=float, default=2e-3)
    s.set_defaults(func=cmd_sweep)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

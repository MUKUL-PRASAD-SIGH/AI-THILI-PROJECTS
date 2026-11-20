"""Run the whole comparison and write the reports.

For every variant (fp32, dynamic, static, int8 ... int2):
  * size: bytes of the saved file (measured)
  * quality: validation loss / perplexity on held-out dialogue, greedy agreement with the
    float model, share of generated words that are real corpus words, sample replies
  * speed: per-operation and whole-reply timing on this CPU (fp32 / dynamic / static, the
    variants with real int8 kernels) and simulated per-token latency on target devices

Outputs: reports/results.json, reports/quantization.md, reports/samples.md, reports/tradeoff.png
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from app import llm, quantize, simulate, storage, train

PROMPTS = [
    ("BLACKSMITH", "Can you repair my sword?"), ("BLACKSMITH", "How much for a shield?"),
    ("INNKEEPER", "Do you have a room?"), ("INNKEEPER", "Heard any rumours?"),
    ("GUARD", "Is the road safe?"), ("GUARD", "Can I enter the town?"),
    ("MERCHANT", "What are you selling?"), ("MERCHANT", "That's too expensive."),
    ("HEALER", "I'm hurt."), ("HEALER", "Do you sell potions?"),
    ("BARD", "Play me a song."), ("BARD", "Tell me a legend."),
    ("ELDER", "Do you have a task for me?"), ("ELDER", "It's done."),
    ("FERRYMAN", "How much to cross?"), ("FERRYMAN", "Is the river safe?"),
    # wordings that are not in the corpus
    ("GUARD", "Why are there so many soldiers today?"), ("MERCHANT", "Do you have anything for a long journey?"),
    ("HEALER", "My friend was bitten by a snake."), ("BARD", "Sing about the dragon."),
]
BITS = {"fp32": 32, "dynamic": 8, "static": 8, **{f"int{b}": b for b in quantize.BIT_WIDTHS}}
REAL_INT8 = ("fp32", "dynamic", "static")


def word_validity(replies: list[str], vocab: set[str]) -> float:
    words = [w for r in replies for w in re.findall(r"[a-z']+", r.lower())]
    return sum(w in vocab for w in words) / max(1, len(words))


def run(log=print, samples_per_prompt: int = 2, op_repeats: int = 30, prompts=None, max_new_tokens: int = 140,
        bench_tokens: int = 60) -> dict:
    prompts = prompts or PROMPTS
    threads = llm.set_threads()
    model, tok, history = llm.load_checkpoint()
    tok_, train_ids, val_ids = train.load_data()
    assert tok_.chars == tok.chars, "corpus changed since training: retrain"
    vocab = set(re.findall(r"[a-z']+", Path(train.corpus.CORPUS_FILE).read_text(encoding="utf-8").lower()))
    bs = model.cfg.block_size

    log("building and saving quantized variants...")
    variants = quantize.build_all(model, tok, train_ids, log=log)
    breakdown = quantize.size_breakdown(model)

    log("measuring quality...")
    greedy_fp32 = llm.npc_replies(model, tok, prompts, max_new_tokens, temperature=0)
    results: dict[str, dict] = {}
    samples: dict[str, list[list[str]]] = {}
    for name, v in variants.items():
        m = v["model"]
        loss = train.eval_loss(m, val_ids, bs)
        greedy = llm.npc_replies(m, tok, prompts, max_new_tokens, temperature=0)
        sampled = [llm.npc_replies(m, tok, prompts, max_new_tokens, temperature=0.8, top_k=20, seed=s)
                   for s in range(samples_per_prompt)]
        samples[name] = [greedy] + sampled
        results[name] = {
            "kind": v["kind"], "file": v["file"], "bits": BITS[name], "bytes": v["bytes"],
            "val_loss": round(loss, 4), "perplexity": round(math.exp(loss), 3),
            "greedy_same_as_fp32": sum(a == b for a, b in zip(greedy, greedy_fp32)) / len(prompts),
            "word_validity": round(word_validity([r for s in samples[name] for r in s], vocab), 4),
        }
        log(f"{name:>8}: val loss {loss:.4f}  greedy same as fp32 {results[name]['greedy_same_as_fp32']:.0%}")

    fp32_bytes = results["fp32"]["bytes"]
    for r in results.values():
        r["size_ratio"] = round(fp32_bytes / r["bytes"], 2)
        r["val_loss_delta"] = round(r["val_loss"] - results["fp32"]["val_loss"], 4)

    log("timing operations and generation on this CPU...")
    real = {k: variants[k]["model"] for k in REAL_INT8}
    ops = simulate.benchmark_ops(real, bs, repeats=op_repeats)
    gen = simulate.benchmark_generation(real, tok, prompts[:8], new_tokens=bench_tokens)
    for k, g in gen.items():
        results[k]["measured_tokens_per_s"] = g["tokens_per_s"]
    log("simulating devices...")
    devices = simulate.simulate_devices(model, BITS)

    out = {"threads": threads, "params": model.num_params(), "model_config": model.cfg.to_dict(),
           "train": {"best_val_loss": history.get("best_val_loss"), "train_seconds": history.get("train_seconds"),
                     "steps": history.get("train_config", {}).get("steps")},
           "fp32_breakdown_bytes": breakdown, "variants": results, "ops": ops, "generation": gen,
           "devices": simulate.devices_table(), "device_sim": devices}
    storage.write_json(storage.REPORTS / "results.json", out)
    write_samples(samples, prompts, storage.REPORTS / "samples.md")
    plot(out, storage.REPORTS / "tradeoff.png")
    write_markdown(out, storage.REPORTS / "quantization.md")
    log(f"wrote {storage.display(storage.REPORTS / 'quantization.md')}")
    return out


# --- writing -------------------------------------------------------------------------------

def _mb(b: float) -> str:
    return f"{b / 1e6:.3f} MB"


def write_samples(samples: dict[str, list[list[str]]], prompts, path: Path) -> None:
    lines = ["# Sample NPC replies per model variant", "",
             "Same prompts and same random seeds for every variant. Row 1 of each block is greedy "
             "(temperature 0); the others are sampled at temperature 0.8, top-k 20.", ""]
    names = list(samples)
    for i, (role, line) in enumerate(prompts):
        novel = " *(wording not in the corpus)*" if (role, line) in PROMPTS[16:] else ""
        lines += [f"### PLAYER: {line}  →  {role}{novel}", "", "| Variant | Greedy | " + " | ".join(f"Sample {k}" for k in range(1, len(samples[names[0]]))) + " |",
                  "|---|---|" + "---|" * (len(samples[names[0]]) - 1)]
        for n in names:
            cells = [s[i].replace("|", "\\|") or "*(empty)*" for s in samples[n]]
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_markdown(out: dict, path: Path) -> None:
    v = out["variants"]
    b = out["fp32_breakdown_bytes"]
    L = ["# Quantization report", "",
         f"Model: {out['params']:,} parameters ({out['model_config']['n_layer']} layers, d={out['model_config']['n_embd']}, "
         f"context {out['model_config']['block_size']}, {out['model_config']['vocab_size']}-character vocabulary). "
         f"Timings on this machine used {out['threads']} CPU threads.", "",
         "## Size and quality", "",
         "| Variant | Method | File size | Smaller by | Val loss | Δ loss | Perplexity | Greedy = fp32 | Real words |",
         "|---|---|---|---|---|---|---|---|---|"]
    for n, r in v.items():
        L.append(f"| **{n}** | {r['kind']} | {_mb(r['bytes'])} | {r['size_ratio']}x | {r['val_loss']:.4f} | "
                 f"{r['val_loss_delta']:+.4f} | {r['perplexity']:.3f} | {r['greedy_same_as_fp32']:.0%} | {r['word_validity']:.1%} |")
    L += ["", "*Val loss*: next-character cross-entropy on held-out dialogue (lower is better). *Greedy = fp32*: "
          "share of the 20 test prompts where the greedy reply is character-for-character the float model's. "
          "*Real words*: share of generated words that occur in the corpus.", "",
          f"Where the float32 bytes are: Linear weights {_mb(b['linear_weights'])} "
          f"({b['linear_weights'] / b['total']:.0%}), biases {_mb(b['linear_biases'])}, embeddings "
          f"{_mb(b['embeddings'])}, LayerNorms {_mb(b['layernorms'])}. Only Linear weights are quantized, so even "
          f"1-bit weights could not go below {_mb(b['total'] - b['linear_weights'])}.", "",
          "## Operations on float vs quantized weights (this CPU)", "",
          "Median time of one `y = x Wᵀ + b` per Linear shape. *Decode* = 1 token (what each generated "
          "character costs), *prefill* = a full context window.", "",
          "| Layer | Shape | Per model | Tokens | fp32 | dynamic int8 | static int8 | dynamic speed-up | static speed-up |",
          "|---|---|---|---|---|---|---|---|---|"]
    for o in out["ops"]:
        L.append(f"| {o['layer']} | {o['in']}→{o['out']} | {o['per_model']} | {o['tokens']} | {o['fp32_us']:.1f} µs | "
                 f"{o['dynamic_us']:.1f} µs | {o['static_us']:.1f} µs | {o['fp32_us'] / o['dynamic_us']:.2f}x | "
                 f"{o['fp32_us'] / o['static_us']:.2f}x |")
    g = out["generation"]
    L += ["", "Whole replies (greedy, 60 characters each, 8 prompts, batch 1):", "",
          "| Variant | Tokens/s | ms per token | vs fp32 |", "|---|---|---|---|"]
    for n, r in g.items():
        L.append(f"| {n} | {r['tokens_per_s']} | {r['ms_per_token']} | {r['tokens_per_s'] / g['fp32']['tokens_per_s']:.2f}x |")
    L += ["", "## Simulated devices", "",
          "Roofline estimate per generated token: max(bytes read ÷ bandwidth, operations ÷ throughput), "
          "int8 throughput for ≤8-bit weights. Device figures are assumptions (edit `DEVICES` in `app/simulate.py`):", "",
          "| Device | Bandwidth | fp32 | int8 | Model budget | Notes |", "|---|---|---|---|---|---|"]
    for d in out["devices"]:
        L.append(f"| {d['name']} | {d['bandwidth_gbs']} GB/s | {d['fp32_gflops']} GFLOP/s | {d['int8_gops']} GOP/s | "
                 f"{d['budget_mb']} MB | {d['note']} |")
    for dev in [d["name"] for d in out["devices"]]:
        L += ["", f"**{dev}**", "", "| Variant | Model size | Fits | ms/token | Tokens/s | 60-char reply | Bound by |",
              "|---|---|---|---|---|---|---|"]
        for r in [r for r in out["device_sim"] if r["device"] == dev]:
            L.append(f"| {r['variant']} | {r['model_mb']} MB | {'yes' if r['fits'] else '**no**'} | {r['ms_per_token']} | "
                     f"{r['tokens_per_s']:,} | {r['reply_ms']} ms | {r['bound']} |")
    L += ["", "![Size, quality and simulated speed](tradeoff.png)", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def plot(out: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
    blue, orange, gray = "#2a78d6", "#eb6834", "#8a8984"
    v = out["variants"]
    fam = {n: (gray if n == "fp32" else blue if n in ("dynamic", "static") else orange) for n in v}

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor=surface,
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})
    for ax in axes:
        ax.set_facecolor(surface)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)

    ax = axes[0]
    # label offsets (points): each label sits right next to its own point; dynamic/int8 share a spot
    offsets = {"fp32": (0, 10, "center"), "dynamic": (0, 11, "center"), "static": (10, -3, "left"),
               "int8": (4, -16, "left"), "int6": (0, -16, "center"), "int4": (0, 10, "center"),
               "int3": (-9, -3, "right"), "int2": (9, -3, "left")}
    for n, r in v.items():
        x, y = r["bytes"] / 1e6, r["val_loss"]
        torch_ptq = n in ("dynamic", "static")
        # squares (PyTorch) drawn larger and underneath, so a circle at the same spot stays visible
        ax.scatter(x, y, s=110 if torch_ptq else 50, marker="s" if torch_ptq else "o", color=fam[n],
                   edgecolor=surface, linewidth=1.5, zorder=3 if torch_ptq else 4)
        dx, dy, ha = offsets.get(n, (7, 4, "left"))
        ax.annotate(n, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=ink, ha=ha)
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.15, 0.2, 0.3, 0.5, 1.0, 1.5]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_ylim(0.14, 2.0)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.3, 0.5, 1.0, 2.0, 3.0]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_xlim(0.28, 4.2)
    ax.set_xlabel("File size (MB, log scale)", color=muted)
    ax.set_ylabel("Validation loss, log scale (lower is better)", color=muted)
    ax.set_title("Size vs dialogue quality", loc="left", color=ink, fontsize=11, fontweight="bold")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=m, ls="", color=c, label=l, markersize=8)
                       for c, m, l in ((gray, "o", "float32"), (blue, "s", "PyTorch int8 PTQ"),
                                       (orange, "o", "from-scratch k-bit"))],
              frameon=False, fontsize=9, labelcolor=ink, loc="upper right")

    for ax, dev in zip(axes[1:], (out["devices"][1], out["devices"][2])):
        rows = [r for r in out["device_sim"] if r["device"] == dev["name"]]
        names = [r["variant"] for r in rows][::-1]
        vals = [r["tokens_per_s"] for r in rows][::-1]
        fits = [r["fits"] for r in rows][::-1]
        bars = ax.barh(names, vals, height=0.6, color=[fam[n] if f else surface for n, f in zip(names, fits)],
                       edgecolor=[fam[n] for n in names], linewidth=1.5, hatch=[None if f else "///" for f in fits])
        for bar, val, f in zip(bars, vals, fits):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f"  {val:,.0f}" + ("" if f else "  doesn't fit"), va="center", fontsize=9, color=ink)
        ax.set_xlim(0, max(vals) * 1.45)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Simulated tokens per second", color=muted)
        short = dev["name"].split(" (")[0]
        ax.set_title(f"{short}: {dev['budget_mb']:g} MB budget", loc="left", color=ink, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

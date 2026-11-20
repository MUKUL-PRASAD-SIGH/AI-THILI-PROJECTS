"""Merge the trained LoRA adapter into GPT-2's weights for deployment.

``W' = W + (alpha/r) B A`` for every adapted layer, then the LoRA wrappers are
removed. The merged model is a standard GPT-2 checkpoint: it loads with plain
``AutoModelForCausalLM.from_pretrained`` (no LoRA code), has the same parameter
count as the original, and runs without the extra adapter matmuls.

``merge_and_save`` checks the merge is exact by comparing logits from the
adapted and merged model on every FAQ prompt, and times both.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from app import core, llm, lora, storage


@torch.no_grad()
def prompt_logits(model, tokenizer, questions: list[str]) -> list[torch.Tensor]:
    device = next(model.parameters()).device
    out = []
    for q in questions:
        ids = torch.tensor([tokenizer.encode(core.format_prompt(q))], device=device)
        out.append(model(input_ids=ids).logits[0].float().cpu())
    return out


def _time_generation(model, tokenizer, questions: list[str]) -> float:
    start = time.perf_counter()
    llm.generate_answers(model, tokenizer, questions, max_new_tokens=40, batch_size=1)
    return (time.perf_counter() - start) / len(questions)


def _dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def _display(path: Path) -> str:
    """Path relative to the project when possible, so reports don't leak local folders."""
    try:
        return Path(path).resolve().relative_to(storage.PROJECT_ROOT).as_posix()
    except ValueError:
        return Path(path).name


def merge_and_save(adapter_path: str | Path | None = None, out_dir: str | Path | None = None,
                   n_check: int = 60, n_timing: int = 8, log=print) -> dict:
    adapter_path = Path(adapter_path or llm.adapter_dir())
    out_dir = Path(out_dir or llm.merged_dir())
    faqs = core.load_faqs()
    _, held_out = core.split_examples(faqs)
    questions = [e.question for e in held_out][:n_check]

    model, tok = llm.load_adapted(adapter_path, device="cpu")
    n_layers = len(lora.lora_layers(model))
    adapted_params = lora.count_parameters(model)[1]
    before = prompt_logits(model, tok, questions)
    adapted_answers = llm.generate_answers(model, tok, questions)
    t_adapter = _time_generation(model, tok, questions[:n_timing])

    lora.merge_lora(model)
    after = prompt_logits(model, tok, questions)
    merged_answers = llm.generate_answers(model, tok, questions)
    t_merged = _time_generation(model, tok, questions[:n_timing])

    max_diff = max(float((a - b).abs().max()) for a, b in zip(before, after))
    same_answers = sum(a == b for a, b in zip(adapted_answers, merged_answers))
    log(f"merged {n_layers} LoRA layers; max |logit difference| over {len(questions)} prompts = {max_diff:.2e}")
    log(f"identical greedy answers: {same_answers}/{len(questions)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    tok.save_pretrained(out_dir)

    reloaded, _ = llm.load_merged(out_dir, device="cpu")
    reload_diff = max(float((a - b).abs().max()) for a, b in zip(after, prompt_logits(reloaded, tok, questions[:5])))
    report = {
        "adapter_dir": _display(adapter_path),
        "merged_dir": _display(out_dir),
        "lora_layers_merged": n_layers,
        "params_with_adapter": adapted_params,
        "params_merged": lora.count_parameters(reloaded)[1],
        "lora_modules_left": len(lora.lora_layers(reloaded)),
        "max_logit_diff": max_diff,
        "reload_max_logit_diff": reload_diff,
        "identical_answers": same_answers,
        "n_checked": len(questions),
        "sec_per_answer_adapter": round(t_adapter, 3),
        "sec_per_answer_merged": round(t_merged, 3),
        "adapter_size_mb": round(_dir_size_mb(adapter_path), 2),
        "merged_size_mb": round(_dir_size_mb(out_dir), 1),
    }
    storage.write_json(out_dir / "merge_report.json", report)
    log(f"saved merged model to {out_dir} ({report['merged_size_mb']} MB); "
        f"{t_adapter:.3f}s/answer with adapter vs {t_merged:.3f}s merged")
    return report

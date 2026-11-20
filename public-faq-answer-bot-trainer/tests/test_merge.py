"""Merging on the tiny GPT-2: exact outputs, plain checkpoint, reloadable without LoRA code."""
import torch
from transformers import AutoModelForCausalLM

from app import llm, lora, merge


def test_merged_checkpoint_matches_adapter_and_reloads(tiny_gpt2, tokenizer, tmp_path):
    n_params = lora.count_parameters(tiny_gpt2)[1]
    lora.inject_lora(tiny_gpt2, lora.LoRAConfig(r=4, alpha=8, dropout=0.0))
    with torch.no_grad():
        for m in lora.lora_layers(tiny_gpt2).values():
            m.lora_B.normal_(0, 0.05)
    tiny_gpt2.eval()
    questions = ["How do I renew?", "When is the pool open?"]
    adapted = merge.prompt_logits(tiny_gpt2, tokenizer, questions)
    adapted_answers = llm.generate_answers(tiny_gpt2, tokenizer, questions, max_new_tokens=8)

    lora.merge_lora(tiny_gpt2)
    tiny_gpt2.save_pretrained(tmp_path, safe_serialization=True)
    reloaded = AutoModelForCausalLM.from_pretrained(tmp_path).eval()

    assert lora.count_parameters(reloaded)[1] == n_params
    assert not lora.lora_layers(reloaded)
    for a, b in zip(adapted, merge.prompt_logits(reloaded, tokenizer, questions)):
        assert torch.allclose(a, b, atol=1e-4)
    assert llm.generate_answers(reloaded, tokenizer, questions, max_new_tokens=8) == adapted_answers

"""A tiny randomly initialised GPT-2 and a character tokenizer, so LoRA tests need no download."""
import pytest
import torch


class CharTokenizer:
    """Byte-level stand-in for the GPT-2 tokenizer: encode/decode/eos plus batch __call__."""

    eos_token_id = 0
    pad_token_id = 0
    eos_token = "\x00"
    pad_token = "\x00"
    padding_side = "left"

    def encode(self, text):
        return [min(b, 254) + 1 for b in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=True):
        ids = [int(i) for i in ids if not (skip_special_tokens and int(i) == 0)]
        return bytes(i - 1 for i in ids).decode("utf-8", errors="ignore")

    def __call__(self, texts, return_tensors="pt", padding=True):
        enc = [self.encode(t) for t in texts]
        width = max(map(len, enc))
        ids = torch.zeros(len(enc), width, dtype=torch.long)
        mask = torch.zeros(len(enc), width, dtype=torch.long)
        for i, e in enumerate(enc):
            ids[i, width - len(e):] = torch.tensor(e)
            mask[i, width - len(e):] = 1

        class Enc(dict):
            def to(self, device):
                return Enc({k: v.to(device) for k, v in self.items()})

        return Enc(input_ids=ids, attention_mask=mask)


@pytest.fixture
def tokenizer():
    return CharTokenizer()


@pytest.fixture
def tiny_gpt2():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    cfg = GPT2Config(vocab_size=256, n_positions=256, n_embd=32, n_layer=2, n_head=2,
                     bos_token_id=0, eos_token_id=0, initializer_range=0.3)
    return GPT2LMHeadModel(cfg).eval()

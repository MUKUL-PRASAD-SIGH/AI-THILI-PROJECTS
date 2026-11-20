# AI Product Describer

A command-line tool that writes product descriptions for a small candle shop.
It trains its **own GPT from scratch** on the shop's existing product texts:
a byte-pair tokenizer, token + position embeddings, hand-written causal
self-attention and transformer blocks. It uses no pretrained weights and no API key.

```
$ python -m app.cli describe --name "Midnight Fig" --notes "black fig, cedarwood" --size "10 oz" --length medium --variants 2 --seed 42

Midnight Fig (medium)
---------------------
1. Ripe black fig takes the lead in Midnight Fig, softened by smooth cedarwood. Lush and fruity, like summer in a jar. Light it as you run a warm bath. Our 10 oz ceramic bowl offers up to 60 hours of fragrance.

2. Ripe black fig meets dry cedarwood in this moody, grown-up candle. It is rich without being sweet, and it fills a living room in under an hour. Expect around 60 hours of burn time from this 10 oz candle.
```

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional: no keys needed, only MODEL_DIR / DEVICE
python -m app.cli hello
```

Train the model (the corpus is already in `data/corpus.txt`):

```bash
python -m app.cli train                 # ~1M-parameter GPT, 1,500 steps
python -m app.cli train --steps 300     # quicker, rougher model for a first try
```

Generate descriptions:

```bash
# one product, from flags
python -m app.cli describe --name "Sea Salt Morning" --notes "sea salt, lemon" --size "6 oz"

# desired length, several variants, saved to a text file
python -m app.cli describe --name "Pine Cabin" --notes "pine, smoked birch" \
    --length long --variants 3 --out my_descriptions.txt

# no --name: the tool asks for each detail and offers to save
python -m app.cli describe
```

Other commands:

| Command | What it does |
|---|---|
| `build-corpus [--n 2400]` | rebuild `data/corpus.txt` from the raw listings + catalog generator |
| `tokenize "text"` | show how the trained BPE tokenizer splits some text |
| `describe --temperature 0.6 --top-k 20 --seed 42` | tune sampling; a seed makes output repeatable |

Run the tests with `pytest`. They train a tiny model in a few seconds, so they
don't need a trained `models/` folder.

## How it works

| Milestone | Where | What |
|---|---|---|
| 1. Corpus | `app/corpus.py`, `data/` | 19 hand-written listings (`data/raw/seed_descriptions.txt`) are cleaned: HTML, prices, "SHOP NOW", smart quotes, `!!!` and a duplicate listing are removed. Twenty listings are too few to train a GPT, so a seeded catalog generator adds 2,400 more products. It builds them from a scent library of 57 notes in 8 families with many sentence patterns. Each product becomes one record: `<\|product\|>name: … \| notes: … \| size: … \| length: short/medium/long` → `<\|description\|>…<\|end\|>` |
| 2. Tokenizer + embeddings | `app/tokenizer.py`, `app/model.py` | Byte-level BPE trained from scratch, using a GPT-2-style regex pre-split and atomic special tokens. It saves and loads as JSON. With 1,024 tokens, a candle description is about 4 bytes per token (e.g. `' cedarwood'` is one token). `Embeddings` adds a learned position vector to each token vector. |
| 3. Attention + blocks | `app/model.py` | `CausalSelfAttention` is written out by hand: Q/K/V projection, split into heads, scaled dot product, lower-triangular mask, softmax, re-join. `Block` = pre-LayerNorm → attention → residual → LayerNorm → MLP (GELU) → residual. `GPT` stacks the blocks with a tied output head. Tests check that changing future tokens never changes earlier outputs. |
| 4. Training | `app/train.py` | 90/10 train/validation split by product. Windows mostly start at a product header, so the model sees whole header → description pairs. Training uses AdamW, warm-up + cosine LR, gradient clipping, and keeps the checkpoint with the best validation loss. Writes `models/describer.pt`, `models/tokenizer.json` and `models/train_log.json`. |
| 5. Generation | `app/model.py`, `app/llm.py` | Autoregressive sampling with temperature and top-k. It stops at `<\|end\|>` and trims to the last full sentence. The `length` field in the header steers the model, backed by a per-length token budget. Variants are de-duplicated. |
| 6. CLI | `app/cli.py` | `train`, `describe` (flags or interactive), `tokenize`, `build-corpus`. Stretch goals are all in: `--length`, `--variants`, `--out`. |

## Results

Training run on a laptop CPU: 940,800 parameters, 4 layers × 4 heads, 128-dim
embeddings, 128-token context, 1,024-token BPE vocabulary, 1,500 steps of
32 windows, about 16 minutes.

| step | train loss | val loss |
|---:|---:|---:|
| 0 | 6.960 | 6.960 |
| 150 | 1.438 | 1.467 |
| 300 | 0.799 | 0.840 |
| 600 | 0.464 | 0.518 |
| 900 | 0.382 | 0.439 |
| 1500 | 0.331 | **0.414** |

Step 0 is ln(1024) ≈ 6.93, i.e. guessing at random. Validation loss is
measured on products the model never saw, and it stays close to the training
loss, so the model has not simply memorised the corpus.

More samples (`--seed` fixed so you can reproduce them):

```
Sea Salt Morning (short)
1. Crisp sea salt and zesty lemon peel come together like an open window by the coast. A fresh, light scent that clears the air.

Chai Ember (long)
Warm cardamom, zingy ginger root and golden honey make this candle feel glowing from the very first
light. It makes any room feel like a gathering. Made for dining tables, it fills the space in under
an hour. Vegan, cruelty-free and poured by hand. The 12 oz hand-thrown ceramic bowl burns for up to
70 hours. Wrap it up for anyone who loves a cozy night in.

$ python -m app.cli tokenize "Hand-poured soy candle with dry cedarwood"
H | and | - | poured |  soy |  candle |  with |  d | ry |  cedarwood
10 tokens for 41 bytes
```

The model takes the notes, size and burn time (8 oz → 50 h, 12 oz → 70 h, …)
from the header. It also follows the requested length: short gives 2 sentences
and long gives 5–6.

## Honest limits

* Most of the corpus comes from a template generator, so the model mostly
  learns that house style. To make it sound like *your* shop, put your own
  listings in `data/raw/seed_descriptions.txt` (same `Name/Notes/Size` format),
  run `build-corpus` with a smaller `--n`, and retrain.
* A 1M-parameter model trained for minutes on a laptop CPU makes small
  mistakes. In one sample it renamed "Sea Salt Morning" to "Sea Salt Coast",
  and it sometimes pairs a scent with a mood from another family. Seed
  listings are repeated in training, so a familiar product can come back
  close to its original text (variant 2 above). Generate a few `--variants`
  and pick one.
* Notes it has never seen (e.g. "durian") get spelled out by the byte-level
  tokenizer and still appear in the header. The model handles them less
  gracefully than familiar notes.

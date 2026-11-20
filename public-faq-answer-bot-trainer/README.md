# Public FAQ Answer Bot Trainer

A Streamlit app that lets public service administrators fine-tune **GPT-2** on their
departments' FAQ guidelines so it answers residents' questions, which cuts down on
repeat enquiries. The fine-tuning uses **LoRA written from scratch** in plain PyTorch
(`app/lora.py`, not the `peft` library). Only 1.9% of the model's parameters are trained.
After training, the adapter is **merged** into the weights to give one standard GPT-2
checkpoint for deployment.

Everything runs locally on a laptop CPU. No API key or hosted model is needed.

```
$ python -m app.cli ask "Can I replace my roof shingles without a permit?" "What ID is required to apply for a driver's licence?"
[merged] Q: Can I replace my roof shingles without a permit?
       A: A permit is required to replace more than 25 percent of a roof. Repairs smaller than that do not need a permit.

[merged] Q: What ID is required to apply for a driver's licence?
       A: Bring one proof of identity such as a passport or birth certificate, two proofs of Riverton address, and your Social Security card.
```

Neither question was in the training data: they are held-out paraphrases.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env            # optional: no keys, only BASE_MODEL / MODEL_DIR / DEVICE
```

Train, check and merge (GPT-2, ~500 MB, downloads from Hugging Face on first use):

```bash
python -m app.cli train         # LoRA r=16 on all GPT-2 projections, 20 epochs -> models/adapter (~25 min on CPU)
python -m app.cli evaluate      # scores base GPT-2, a lookup baseline and the LoRA model -> reports/evaluation.md
python -m app.cli merge         # folds the adapter into the weights -> models/merged
python -m app.cli ask "When is my water bill due?"
```

Then open the app:

```bash
streamlit run app/main.py
```

| Tab | What it does |
|---|---|
| **Ask the bot** | Ask a question, or pick a held-out example by department. The answer is shown next to the closest approved FAQ, with a ✅/❌ for each key fact and a warning for any number not in the approved answer. Administrators can rate it from 1 to 5. |
| **FAQ dataset** | Browse and filter the FAQs by department. |
| **Train** | Pick rank, alpha, epochs, learning rate and target layers, then train a new adapter and watch the loss curve. |
| **Evaluation** | Run the evaluation and read the metrics, plus every held-out answer (with a filter for the wrong ones) and the LoRA settings comparison. |
| **Merge for deployment** | Merge the adapter and see the checks: logit difference, identical answers, parameter count, speed. |
| **Feedback** | Average rating per model, plus the low-rated answers to fix in the dataset before the next training run. |

The sidebar switches between the **merged** model, **base + adapter** (unmerged) and **plain GPT-2**.

Tests: `pytest`. They use a tiny randomly initialised GPT-2, so they need no download and
no trained model (32 tests, about 40 s). CI runs them on every push.

## How it works

### 1. Dataset (`data/faqs.jsonl`, `app/core.py`)

There are 60 FAQs from 10 departments of the fictional **City of Riverton**: Motor Vehicles,
Water and Sewer, Waste and Recycling, Property Tax, Building Permits, Vital Records, Parks
and Recreation, Public Library, General, and Out of scope. The city is made up on purpose, so the bot never
states a real government's fees or rules. Each record has:

```json
{"id": "dmv-03", "department": "Motor Vehicles",
 "question": "How much does it cost to register a car?",
 "paraphrases": ["What is the fee for vehicle registration?", "What does car registration cost each year?",
                 "How much will registering my vehicle cost?"],
 "answer": "Vehicle registration costs $48 per year for a passenger car and $30 per year for a motorcycle.",
 "facts": ["$48", "$30", "motorcycle"]}
```

`facts` are the parts of the answer a resident acts on, such as fees, deadlines, phone numbers
and places. The loader rejects a record if one of its facts isn't in its answer, if two FAQs
share a wording, or if a line is malformed (it reports the line number). The **last
paraphrase of every FAQ is held out**. The model trains on the other 180 wordings and is
tested on 60 it has never seen.

The prompt format is:

```
Riverton City Services FAQ
Question: <question>
Answer: <answer><|endoftext|>
```

### 2. LoRA from scratch (`app/lora.py`, `app/train.py`)

`LoRALinear` wraps a frozen layer `W` and adds a trainable low-rank update:

```
h = W x + (alpha / r) · B A x        A: r × in (Kaiming init)   B: out × r (zeros)
```

Because `B` starts at zero, the adapted model is exactly GPT-2 at step 0 (there's a test for this).
GPT-2 stores its projections as `Conv1D`, whose weight is transposed compared with `nn.Linear`, and
`LoRALinear` handles both layouts. `inject_lora` swaps the target layers (matched by name
suffix) for wrappers and freezes everything else. With the defaults (r=16, alpha=32, on
`attn.c_attn`, `attn.c_proj`, `mlp.c_fc` and `mlp.c_proj` in all 12 blocks), that is 48 layers and
**2,359,296 trainable parameters out of 126.8M (1.86%)**. The adapter file is 9.4 MB.

Training computes the loss **only on the answer tokens and EOS**: prompt tokens get label -100.
So the model learns to answer and to stop, rather than to repeat questions. It uses AdamW at
lr 2e-3 with warm-up and a cosine schedule, gradient clipping, and keeps the adapter with the
lowest held-out loss.

### 3. Evaluation (`app/evaluate.py`): [full report](reports/evaluation.md)

Each answer is scored against the approved answer for its FAQ:

| Metric | Meaning |
|---|---|
| **Fact recall** | Share of the FAQ's key facts present in the answer. This is the main accuracy number. |
| **All facts** | The answer has every key fact. |
| **Relevant** | Out of all 60 approved answers, the generated one is closest to the right FAQ, so the bot answered the question asked rather than a neighbouring one. |
| **Unsupported numbers** | The answer states a fee, time, phone number or count that isn't in the approved answer, which makes it a likely hallucination. Lower is better. |
| Token F1, ROUGE-L, exact match | Standard overlap with the reference. |

Three systems are compared on the same questions with greedy decoding. **GPT-2 with no
fine-tuning** gets the same prompt. The **word-overlap lookup** is a simple baseline: it returns
the stored answer of the training question with the most words in common.

**Held-out paraphrases (60 questions never seen in training):**

| System | Fact recall | All facts | Relevant | Unsupported numbers ↓ | ROUGE-L |
|---|---|---|---|---|---|
| GPT-2, no fine-tuning | 1.4% | 0.0% | 13.3% | 16.7% | 0.144 |
| Word-overlap lookup | 58.8% | 55.0% | 55.0% | 41.7% | 0.622 |
| **LoRA GPT-2 (merged)** | **84.6%** | **76.7%** | **83.3%** | **15.0%** | **0.846** |

On the 60 seen questions, the LoRA model gets 100% on every metric. By department, held-out
all-facts accuracy ranges from 87.5% (Motor Vehicles, Water and Sewer, Waste and Recycling) down to 60%
(Vital Records).

The report also lists the **qualitative** evidence: side-by-side answers from all three systems
and every held-out answer the LoRA model got wrong, with the reason. The 14 misses are of two
kinds. Some start on the right FAQ and then pull in a neighbouring FAQ's facts: "Can I order an
extra recycling bin?" got the bulky-pickup fee of $35. Others are correct but incomplete: "When
can an inspector come out?" got only "Inspections are done the next working day." The Ask tab
shows the closest approved answer and marks missing facts, so an administrator can catch both.

**How I got there.** The first run trained on 2 wordings per FAQ for 12 epochs at lr 1e-3
([report](reports/evaluation_v1.md)). It reached only 38.3% all-facts, 60% relevant and 48%
unsupported numbers on held-out questions, and only 73% all-facts even on *seen* questions,
so it was underfitting. Fees and hours came out close but wrong. Adding a third training
paraphrase per FAQ and training for 20 epochs at lr 2e-3 doubled held-out all-facts
accuracy. The held-out set was the same in both runs.

### 4. Merging for deployment (`app/merge.py`)

`merge_lora` computes `W' = W + (alpha/r) · B A` for each adapted layer, then puts the plain
layer back. The result is an ordinary GPT-2 checkpoint that loads with
`AutoModelForCausalLM.from_pretrained("models/merged")`, without any of this project's code.
From [reports/merge_report.json](reports/merge_report.json):

| | Base + adapter | Merged |
|---|---|---|
| Parameters | 126,799,104 | 124,439,808 (same as GPT-2) |
| LoRA modules | 48 | 0 |
| Seconds per answer (CPU, batch 1) | 2.00 | **1.39** |
| Identical greedy answers | | 60 / 60 |
| Max logit difference | | 1.3e-3 (float32 rounding) |

### Stretch goals

- **Administrator feedback:** answers are rated 1–5 in the app and logged to
  `models/feedback.jsonl`. The Feedback tab summarises ratings per model and lists low-rated
  answers to correct in the dataset before the next training run.
- **LoRA configurations:** `python -m app.cli sweep` trains several rank, alpha and
  target-layer settings with the same data, seed and epoch budget, and compares held-out
  accuracy against trainable parameters and training time. Results are in
  [reports/lora_sweep.md](reports/lora_sweep.md).

## Limitations

- **Out-of-scope questions are not handled.** There are only two "I can only answer questions
  about city services" FAQs, and the model answers held-out off-topic questions such as
  "What is a good restaurant near me?" with invented opening hours for a "Riverside Inn" (0% on that group). A real
  deployment needs many more refusal examples, or a similarity threshold that sends unfamiliar
  questions to 311.
- The model sometimes mixes facts from two similar FAQs. Any deployment should show the source FAQ next
  to the answer, as the Ask tab does, and an administrator should review low-rated answers.
- GPT-2 small learns these 60 FAQs, but it doesn't pick up new rules from a guideline document.
  Every new or changed answer has to be added to `data/faqs.jsonl` and the adapter retrained.

## Project layout

```
app/core.py      dataset loading, validation, prompt format, held-out split
app/lora.py      LoRALinear, inject / freeze / save / load / merge (from scratch)
app/train.py     answer-only loss, training loop, best-checkpoint selection
app/llm.py       load base / adapter / merged model, batched generation
app/evaluate.py  metrics, lookup baseline, markdown report
app/merge.py     merge + exactness and speed checks
app/storage.py   JSONL / JSON files, feedback log, MODEL_DIR
app/cli.py       train, ask, evaluate, merge, sweep
app/main.py      Streamlit app
data/faqs.jsonl  60 FAQs, 10 departments
reports/         evaluation, merge and sweep reports (committed)
models/          adapter, merged model, feedback (git-ignored)
```

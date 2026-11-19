# Support Response Generator

A command-line tool that helps customer support agents draft replies faster.
It learns from an **anonymized corpus of past support conversations** and suggests the
**next word** (top-k) or **completes the sentence**, using a **trigram language model with
add-k smoothing and linear interpolation**, written from scratch in plain Python.

```
Draft: Hi Priya,
Next:  [1] thank (71%)  [2] please (2%)  [3] the (2%)
> 1
Draft: Hi Priya, thank
Next:  [1] you (86%)  [2] . (1%)  [3] for (0%)
> :auto
Draft: Hi Priya, thank you for reaching out.
> The refund will
Next:  [1] be (54%)  [2] receive (4%)  [3] make (3%)
> 1
Next:  [1] processed (36%)  [2] shipped (22%)  [3] available (7%)
> :auto
Final response:
Hi Priya, thank you for reaching out. The refund will be processed within 5 to 7 business days.
```

The name the agent types stays in their draft. It is only replaced by `<NAME>` internally
when the model reads the context.

## Quick start

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m app.cli hello             # check it runs
python -m app.cli prepare           # raw transcripts -> anonymized corpus
python -m app.cli train             # fit the trigram model (prints held-out perplexity)
python -m app.cli draft             # interactive drafting
pytest                              # run the tests
```

No API key is needed. The model runs fully offline. `.env` is only used by the optional
`polish` command (see below).

## Commands

| Command | What it does |
|---|---|
| `prepare [--all-speakers]` | Cleans and anonymizes `data/raw_interactions.txt` and writes `data/corpus.txt` (one tokenized sentence per line) and a readable `data/anonymized_interactions.txt`. Uses only agent turns by default. |
| `train [-n 3] [-k 0.01]` | Trains the n-gram model and saves it to `models/ngram.json`. Reports perplexity on a 10% held-out split. |
| `suggest "TEXT" [-k 5]` | Prints the top-k next words for a partial reply, with probabilities. |
| `generate ["TEXT"] [--strategy greedy\|sample] [--temperature 0.7] [--seed N] [--max-words 20]` | Continues the reply until the end of the sentence. |
| `draft [-k 3] [--start "TEXT"]` | Interactive mode (below). |
| `polish "TEXT"` | *Optional.* Asks an LLM (Ollama / Groq / Gemini, set in `.env`) to fix grammar. The text is anonymized before it is sent. |

If no trained model exists yet, `suggest`, `generate` and `draft` train one automatically.

### Interactive draft mode (stretch goals)

| Input | Action |
|---|---|
| `1`, `2`, … | **accept suggestion N and append it to the draft** |
| *Enter* | accept the top suggestion |
| any text | type your own words; suggestions update to match |
| `:more` / `:less` | show more or fewer top-k suggestions |
| `:auto` | auto-complete the current sentence |
| `:undo` / `:clear` | undo the last change / start over |
| `:show`, `:help`, `:done`, `:quit` | |

## How it works

### 1. Anonymized corpus (`app/core.py`: `anonymize`, `tokenize`, `preprocess`)

`data/raw_interactions.txt` holds 28 synthetic support conversations (orders, refunds,
logins, billing, shipping, etc.). They deliberately contain fake personal data, so the
anonymizer has real work to do. Regular expressions replace each kind of PII with a placeholder:

| PII | Example | Placeholder |
|---|---|---|
| Email | `priya.sharma@example.com` | `<EMAIL>` |
| URL | `https://status.example.com` | `<URL>` |
| Card number | `4111 1111 1111 1111`, `card ending 4242` | `<CARD>` |
| Phone | `+1 (415) 555-0132`, `98765 43210` | `<PHONE>` |
| Order id | `#458921`, `ORD-77213` | `<ORDER_ID>` |
| Tracking / account | `1Z999AA10123456784`, `Account 88213490` | `<TRACKING>`, `<ACCOUNT>` |
| Names | `my name is Fatima Khan`, `Hi Rahul,` | `<NAME>` |

The cleaned text is then lowercased and tokenized, keeping words, numbers, contractions,
punctuation and placeholders, and split into sentences. Placeholders stay as single
tokens, so the model still learns patterns like *"Hi `<NAME>`, thank you for…"*. Drafts the
agent types are anonymized the same way before they are used as context.

### 2. N-gram model with smoothing (`NGramModel`)

For each order *i* (unigram, bigram, trigram) the model uses add-k smoothing:

```
P_i(w | h) = (count(h, w) + k) / (count(h) + k·V)
```

These estimates are combined by **linear interpolation**:

```
P(w | h) = Σ λ_i · P_i(w | h_i)
```

Orders whose history never appeared in training are skipped, and the remaining λ's are
renormalised. So an unseen trigram context backs off to the bigram and unigram instead of
falling to a uniform guess. The distribution always sums to 1, which the tests check.

**Tuning.** I compared settings by perplexity on a 10% held-out split and by top-1
accuracy on six common agent phrases:

| λ (uni, bi, tri) | k = 0.1 | k = 0.05 | k = 0.01 |
|---|---|---|---|
| 0.17, 0.33, 0.50 | 18.5 ppl · 3/6 | 14.9 · 5/6 | 11.2 · 6/6 |
| **0.07, 0.29, 0.64** (∝ i²) | 17.6 · 5/6 | 14.0 · 6/6 | **10.4 · 6/6** |

Plain Laplace (k = 1) scores 49.7 perplexity. Large k gives too much probability mass
to the ~240 unseen words in sparse trigram contexts. The defaults are **k = 0.01, λ ∝ i²**.

### 3. Suggestions and generation

* `suggest_next` ranks the whole vocabulary by `P(w | last two words of the current sentence)`
  and returns the top k.
* `generate` continues word by word, either **greedy** (most likely word) or **sample**
  (temperature-scaled sampling from the top-5 words, reproducible with `--seed`). A
  no-repeat-trigram guard stops loops. Output is re-spaced and capitalised for display.

### 4. CLI (`app/cli.py`)

The CLI uses `argparse` subcommands. The interactive loop takes injectable input and output
streams, so the tests can drive it.

## Project layout

```
app/core.py       preprocessing, anonymization, NGramModel, suggest/generate
app/storage.py    read transcripts, write/load corpus, save/load model (JSON)
app/cli.py        command-line interface
app/llm.py        optional LLM "polish" helper (OpenAI-compatible API)
data/             raw synthetic transcripts + generated anonymized corpus
tests/            pytest suite (preprocessing, model, generation, CLI)
```

## Tests and CI

`pytest` runs 30 tests. They check that PII is removed, that probabilities sum to 1, that
smoothing and backoff behave correctly, that serialisation round-trips, that suggestions are
accurate on common phrases, and that the CLI and draft loop work. GitHub Actions runs the
suite on every push (`.github/workflows/support-response-generator.yml` at the repo root).

## Limitations

* The corpus is small and synthetic, so suggestions are only as varied as the 28
  conversations. With so little data a few counts decide everything: after *"your package"* the
  model prefers *"using"* (from "track your package using…") over *"is"*. To use your own data, put it in `data/raw_interactions.txt` (`Customer:` /
  `Agent:` lines) and re-run `prepare` and `train`.
* Name detection relies on context ("my name is …", greetings), so a bare name in the
  middle of a sentence may be missed. A NER model would catch more.
* Add-k with interpolation is simple. Kneser-Ney smoothing would probably lower perplexity further.

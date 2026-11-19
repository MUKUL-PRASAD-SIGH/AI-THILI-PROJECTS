"""Command-line interface for support agents.

    python -m app.cli prepare                 # clean + anonymize raw transcripts
    python -m app.cli train                   # fit the n-gram model
    python -m app.cli suggest "Let me check the" -k 5
    python -m app.cli generate "I'm sorry for the"
    python -m app.cli draft                   # interactive: pick top-k words to build a reply
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from app import storage
from app.core import NGramModel, append_words, complete, current_context, generate, suggest_next


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_model(model_path: str | Path = storage.MODEL_PATH, corpus_path: str | Path = storage.CORPUS_PATH) -> NGramModel:
    """Load the saved model; train (and save) one from the corpus if none exists yet."""
    model_path = Path(model_path)
    if model_path.exists():
        return storage.load_model(model_path)
    model = NGramModel().fit(storage.load_corpus(corpus_path))
    storage.save_model(model, model_path)
    return model


def format_suggestions(suggestions: list[tuple[str, float]]) -> str:
    return "  ".join(f"[{i}] {w} ({p:.0%})" for i, (w, p) in enumerate(suggestions, 1))


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_prepare(args) -> int:
    turns = storage.read_turns(args.input)
    speakers = () if args.all_speakers else ("agent",)
    lines = storage.build_corpus(turns, speakers)
    path = storage.write_corpus(lines, args.output)
    storage.write_anonymized_transcript(turns)
    print(f"Prepared {len(lines)} anonymized sentences from {len(turns)} turns -> {path}")
    return 0


def cmd_train(args) -> int:
    sentences = storage.load_corpus(args.corpus)
    shuffled = sentences[:]
    random.Random(0).shuffle(shuffled)
    cut = max(1, len(shuffled) // 10)
    held_out, train = shuffled[:cut], shuffled[cut:]
    heldout_model = NGramModel(n=args.n, k=args.k).fit(train)
    model = NGramModel(n=args.n, k=args.k).fit(sentences)
    path = storage.save_model(model, args.model)
    print(f"Trained {args.n}-gram model (add-k, k={args.k}) on {len(sentences)} sentences, "
          f"vocab={model.vocab_size} -> {path}")
    print(f"Held-out perplexity (10% split): {heldout_model.perplexity(held_out):.2f}")
    return 0


def cmd_suggest(args) -> int:
    model = get_model(args.model)
    suggestions = suggest_next(model, current_context(args.text), k=args.top_k)
    print(format_suggestions(suggestions))
    return 0


def cmd_generate(args) -> int:
    model = get_model(args.model)
    print(complete(model, args.text, max_words=args.max_words, strategy=args.strategy,
                   temperature=args.temperature, seed=args.seed,
                   stop_at_sentence_end=not args.no_stop))
    return 0


DRAFT_HELP = """\
Build a reply word by word.
  <number>        accept suggestion N        (Enter alone = accept [1])
  any text        type your own words and append them
  :more / :less   show more / fewer suggestions
  :auto           auto-complete the current sentence
  :undo           undo the last change
  :clear          start the draft again
  :show           print the full draft
  :done           finish and print the draft     (:quit exits without printing)
  :help           show this help
"""


def run_draft(model: NGramModel, top_k: int = 3, input_fn=input, output=sys.stdout, start: str = "") -> str:
    """Interactive drafting loop. `input_fn`/`output` are injectable for tests."""
    def say(msg: str = "") -> None:
        print(msg, file=output)

    draft = start
    history: list[str] = []
    say(DRAFT_HELP)
    while True:
        suggestions = suggest_next(model, current_context(draft), k=top_k)
        say(f"\nDraft: {draft or '(empty)'}")
        say("Next:  " + format_suggestions(suggestions))
        try:
            entry = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            say()
            break

        if entry in {":done", ":d"}:
            break
        if entry in {":quit", ":q"}:
            return draft
        if entry == ":help":
            say(DRAFT_HELP)
            continue
        if entry == ":more":
            top_k = min(top_k + 2, 10)
            continue
        if entry == ":less":
            top_k = max(top_k - 2, 1)
            continue
        if entry == ":show":
            say(draft)
            continue
        if entry == ":undo":
            draft = history.pop() if history else draft
            continue

        if entry == "":
            entry = "1"
        if entry.isdigit() and not 1 <= int(entry) <= len(suggestions):
            say(f"Pick a number between 1 and {len(suggestions)}.")
            continue

        history.append(draft)
        if entry == ":clear":
            draft = ""
        elif entry == ":auto":
            draft = append_words(draft, generate(model, current_context(draft)))
        elif entry.isdigit():
            draft = append_words(draft, [suggestions[int(entry) - 1][0]])
        else:
            draft = append_words(draft, entry.split())

    say("\nFinal response:\n" + draft)
    return draft


def cmd_draft(args) -> int:
    run_draft(get_model(args.model), top_k=args.top_k, start=args.start)
    return 0


def cmd_polish(args) -> int:
    from app.llm import polish

    print(polish(args.text))
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="support-response-generator",
                                     description="Next-word suggestions for customer support agents.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the tool runs")

    p = sub.add_parser("prepare", help="clean + anonymize raw transcripts into a corpus")
    p.add_argument("--input", default=str(storage.RAW_PATH))
    p.add_argument("--output", default=str(storage.CORPUS_PATH))
    p.add_argument("--all-speakers", action="store_true", help="also train on customer turns")
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("train", help="train the n-gram model on the prepared corpus")
    p.add_argument("--corpus", default=str(storage.CORPUS_PATH))
    p.add_argument("--model", default=str(storage.MODEL_PATH))
    p.add_argument("-n", type=int, default=3, help="n-gram order (default 3)")
    p.add_argument("-k", type=float, default=0.01, help="add-k smoothing constant (default 0.01)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("suggest", help="show the top-k next words for a partial response")
    p.add_argument("text", help="the response typed so far")
    p.add_argument("-k", "--top-k", type=int, default=3)
    p.add_argument("--model", default=str(storage.MODEL_PATH))
    p.set_defaults(func=cmd_suggest)

    p = sub.add_parser("generate", help="continue a partial response")
    p.add_argument("text", nargs="?", default="", help="the response typed so far")
    p.add_argument("--max-words", type=int, default=20)
    p.add_argument("--strategy", choices=["greedy", "sample"], default="greedy")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-stop", action="store_true", help="keep going past the end of the sentence")
    p.add_argument("--model", default=str(storage.MODEL_PATH))
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("draft", help="interactive mode: build a reply from top-k suggestions")
    p.add_argument("-k", "--top-k", type=int, default=3)
    p.add_argument("--start", default="", help="text to start the draft with")
    p.add_argument("--model", default=str(storage.MODEL_PATH))
    p.set_defaults(func=cmd_draft)

    p = sub.add_parser("polish", help="(optional, needs .env) tidy a finished draft with an LLM")
    p.add_argument("text")
    p.set_defaults(func=cmd_polish)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from support-response-generator!")
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

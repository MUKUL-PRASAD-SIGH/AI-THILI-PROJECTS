"""Command-line interface for the game action predictor."""
from __future__ import annotations

import argparse

from app import storage
from app.core import clean_events, describe, mine_patterns, sequence_stats, to_sequences


def cmd_simulate(args) -> int:
    from app.simulate import simulate_season

    raw = simulate_season(n_games=args.games, seed=args.seed)
    path = storage.write_raw(raw, args.output)
    print(f"Wrote {len(raw):,} raw play-by-play events from {args.games} games -> {path}")
    return 0


def cmd_prepare(args) -> int:
    raw = storage.read_raw(args.input)
    events = clean_events(raw)
    possessions = to_sequences(events)
    path = storage.write_sequences(possessions, args.output)
    stats = sequence_stats(possessions)
    print(f"Cleaned {len(raw):,} raw rows -> {len(events):,} play events "
          f"({len(raw) - len(events):,} timeouts/substitutions/unknown dropped)")
    print(f"{stats['possessions']:,} possessions in {stats['games']} games, "
          f"mean length {stats['mean_length']:.1f} actions, {stats['actions']} distinct actions -> {path}")
    return 0


def patterns_report(pat: dict, top: int = 15) -> str:
    lines = ["# Frequent play patterns", "",
             f"Mined with `thilimarketbasket.mine.frequent_itemsets` over {pat['n_baskets']:,} possession baskets "
             "(actions + ordered transitions + contiguous runs of up to 4 actions + team).", "",
             "## Most frequent play sequences", "", "| # | sequence | support |", "|---|---|---|"]
    lines += [f"| {i} | {describe([s])} | {sup:.1%} |" for i, (s, sup) in enumerate(pat["sequences"][:top], 1)]
    longer = [(s, sup) for s, sup in pat["sequences"] if s.count(">") >= 2][:top]
    lines += ["", "## Most frequent sequences of 3+ actions", "", "| # | sequence | support |", "|---|---|---|"]
    lines += [f"| {i} | {describe([s])} | {sup:.1%} |" for i, (s, sup) in enumerate(longer, 1)]
    lines += ["", "## Team signatures (rule team -> play, by lift)", "",
              "Lift = how many times more often this team's possessions contain the play than the league's.", ""]
    for team in sorted(pat["signatures"]):
        lines += [f"**{team}**", "", "| play | lift | confidence |", "|---|---|---|"]
        lines += [f"| {describe([n])} | {lift:.2f} | {conf:.1%} |" for n, _s, conf, lift in pat["signatures"][team][:5]]
        lines.append("")
    lines += ["## Play -> outcome rules (by lift)", "",
              "\"Possessions that contain this play end with this outcome\" (outcome not part of the play).", "",
              "| play | outcome | support | confidence | lift |", "|---|---|---|---|---|"]
    lines += [f"| {describe([a])} | {c} | {s:.1%} | {cf:.1%} | {lift:.2f} |"
              for a, c, s, cf, lift in pat["outcomes"][:top]]
    return "\n".join(lines) + "\n"


def cmd_mine(args) -> int:
    possessions = storage.read_sequences(args.sequences)
    pat = mine_patterns(possessions, min_support=args.min_support, max_size=args.max_size, max_ngram=args.max_ngram)
    path = storage.write_report("patterns", patterns_report(pat, args.top))
    print(f"{len(pat['itemsets']):,} frequent itemsets (min support {args.min_support:.1%}).")
    print("\nTop play sequences:")
    for s, sup in pat["sequences"][:args.top]:
        print(f"  {sup:6.1%}  {describe([s])}")
    print("\nTeam signatures (highest lift):")
    for team in sorted(pat["signatures"]):
        name, _s, _conf, lift = pat["signatures"][team][0]
        print(f"  {team:<7} {describe([name]):<45} lift {lift:.2f}")
    print(f"\nFull report -> {path}")
    return 0


DEFAULT_BASE = "EleutherAI/pythia-14m"


def set_threads(n: int) -> None:
    """Torch CPU threads. 1 is the safe default: on a laptop short of memory, multi-threaded matrix
    multiplies with a 50k-token vocabulary can crash; a 14M-parameter model trains fine on one core."""
    import torch

    torch.set_num_threads(max(1, n))


def cmd_train(args) -> int:
    from app.core import split_by_game
    from app.lm import ActionLM

    set_threads(args.threads)
    possessions = storage.read_sequences(args.sequences)
    train, test = split_by_game(possessions)
    val = test[:args.val_size]
    if args.from_scratch:
        lm, base = ActionLM.tiny(seed=args.seed), "tiny GPT-2 (random init)"
    else:
        lm, base = ActionLM.load(args.base), args.base
    n_params = sum(x.numel() for x in lm.model.parameters())
    print(f"Continued pretraining of {base} ({n_params / 1e6:.1f}M parameters) on {len(train):,} training "
          f"possessions; validation on {len(val)} possessions from held-out games.")
    history = lm.pretrain(train, val, steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr,
                          warmup=args.warmup, eval_every=args.eval_every, seed=args.seed)
    meta = {"base": base, "steps": args.steps, "batch_size": args.batch_size, "learning_rate": args.lr,
            "seed": args.seed, "train_possessions": len(train), "parameters": n_params}
    out = lm.save(args.out, history, meta)
    print(f"Validation loss {history[0]['val_loss']:.3f} -> {history[-1]['val_loss']:.3f}. Saved -> {out}")
    return 0


# ---------------------------------------------------------------------------
# Milestone 4: evaluation
# ---------------------------------------------------------------------------

def load_lm(path: str, name: str = "language model", window: int | None = None):
    from pathlib import Path

    from app.evaluate import LanguageModel
    from app.lm import ActionLM

    if not Path(path, "config.json").exists():
        raise SystemExit(f"No trained model at {path}. Run `python -m app.cli train` first.")
    return LanguageModel(ActionLM.load(path), name=name, window=window)


def cmd_evaluate(args) -> int:
    import json
    from pathlib import Path

    from app.core import split_by_game
    from app.evaluate import (ItemToItemBaseline, Markov, Oracle, Popularity, evaluate_all, length_table,
                              make_cases, results_table)

    set_threads(args.threads)
    train, test = split_by_game(storage.read_sequences(args.sequences))
    cases = make_cases(test, args.cases, seed=args.seed)
    predictors = [Popularity(train), ItemToItemBaseline(train), Markov(train, 1), Markov(train, 2)]
    lm_names = []
    for spec in [f"language model={args.model}", *args.compare]:
        name, _, path = spec.partition("=")
        predictors.append(load_lm(path, name))
        lm_names.append(name)
    for w in args.windows:
        predictors.append(load_lm(args.model, f"language model, last {w} action(s) only", window=w))
    predictors.append(Oracle())
    print(f"Evaluating {len(predictors)} predictors on {len(cases):,} next-action cases from "
          f"{len({p.game_id for p in test})} held-out games...")
    results = evaluate_all(predictors, cases)

    training = {}
    for name, spec in zip(lm_names, [args.model, *[s.partition("=")[2] for s in args.compare]]):
        info = Path(spec, "training.json")
        if info.exists():
            training[name] = json.loads(info.read_text())
    text = ["# Next-action prediction: evaluation", "",
            f"{len(cases):,} cases from {len({p.game_id for p in test})} held-out games "
            f"(each case: a team and the first actions of a possession; predict the next action). "
            "Ranks and hits@k are computed with `thilimarketbasket.evaluate`.", "",
            "## Overall", "", results_table(results), "",
            "## Top-1 accuracy by number of actions already seen", "", length_table(results), ""]
    if training:
        text += ["## Training runs", "", "| model | base | steps | lr | val loss start -> end |", "|---|---|---|---|---|"]
        for name, t in training.items():
            h = [x for x in t["history"] if x.get("val_loss") is not None]
            text.append(f"| {name} | {t.get('base')} | {t.get('steps')} | {t.get('learning_rate')} | "
                        f"{h[0]['val_loss']:.3f} -> {h[-1]['val_loss']:.3f} |")
        text.append("")
    data = {name: {k: v for k, v in m.items() if k != "ranks"} for name, m in results.items()}
    path = storage.write_report("evaluation", "\n".join(text), {"cases": len(cases), "results": data,
                                                                "training": training})
    print(results_table(results))
    print("\nTop-1 accuracy by actions seen:")
    print(length_table(results))
    print(f"\nReport -> {path}")
    return 0


# ---------------------------------------------------------------------------
# Milestone 5: predictions in the terminal
# ---------------------------------------------------------------------------

def parse_actions(text: str) -> list[str]:
    """'inbound, bring up; Pick & Roll' -> canonical actions. Unknown labels raise a helpful error."""
    import re

    from app.core import normalize_action
    from app.simulate import ACTIONS

    actions = []
    for part in [p.strip() for p in re.split(r"[,;>]|->", text) if p.strip()]:
        action = normalize_action(part)
        if action is None:
            raise ValueError(f"unknown action {part!r}. Known actions: {', '.join(ACTIONS)}")
        actions.append(action)
    return actions


def get_predictor(kind: str, model_path: str):
    """The trained language model, or a Markov-2 counting model (instant, no training needed)."""
    from pathlib import Path

    if kind == "lm" or (kind == "auto" and Path(model_path, "config.json").exists()):
        return load_lm(model_path)
    from app.core import split_by_game
    from app.evaluate import Markov

    train, _ = split_by_game(storage.read_sequences())
    return Markov(train, 2)


def format_predictions(probs, k: int = 5, width: int = 30) -> str:
    from app.simulate import ACTIONS

    order = probs.argsort()[::-1][:k]
    lines = []
    for rank, i in enumerate(order, 1):
        p = float(probs[i])
        bar = "#" * max(1, round(p * width))
        lines.append(f"  {rank}. {ACTIONS[i]:<18} {p:6.1%}  {bar}")
    return "\n".join(lines)


def cmd_predict(args) -> int:
    from app.simulate import TEAMS

    set_threads(args.threads)
    if args.team not in TEAMS:
        raise SystemExit(f"unknown team {args.team!r}; choose from {', '.join(TEAMS)}")
    try:
        history = parse_actions(args.actions)
    except ValueError as err:
        raise SystemExit(str(err))
    predictor = get_predictor(args.model, args.model_path)
    probs = predictor.predict_proba([(args.team, history)])[0]
    print(f"{args.team}: {' -> '.join(history) or '(start of possession)'}")
    print(f"Most likely next actions ({predictor.name}):")
    print(format_predictions(probs, args.top_k))
    return 0


LIVE_HELP = """\
Follow a possession action by action.
  <action>       add what happened (e.g. "pick and roll", "3PT made")
  <number>       accept prediction N as what happened (Enter = accept 1)
  team <name>    switch team (starts a new possession)
  new            start a new possession       undo   remove the last action
  help           this text                    quit   leave
"""


def run_live(predictor, team: str = "Comets", top_k: int = 5, input_fn=input, output=None) -> list[str]:
    """Interactive loop; `input_fn` and `output` are injectable for tests. Returns the last possession."""
    import sys

    from app.simulate import TEAMS

    out = output or sys.stdout

    def say(msg: str = "") -> None:
        print(msg, file=out)

    history: list[str] = []
    say(LIVE_HELP)
    while True:
        probs = predictor.predict_proba([(team, history)])[0]
        order = [int(i) for i in probs.argsort()[::-1][:top_k]]
        say(f"\n{team}: {' -> '.join(history) or '(start of possession)'}")
        say(format_predictions(probs, top_k))
        try:
            entry = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        low = entry.lower()
        if low in {"quit", "q", "exit"}:
            break
        if low == "help":
            say(LIVE_HELP)
        elif low == "new":
            history = []
        elif low == "undo":
            history = history[:-1]
        elif low.startswith("team"):
            name = entry[4:].strip().capitalize()
            if name in TEAMS:
                team, history = name, []
            else:
                say(f"Unknown team. Choose from: {', '.join(TEAMS)}")
        elif entry == "" or entry.isdigit():
            choice = int(entry or 1)
            if 1 <= choice <= len(order):
                from app.simulate import ACTIONS

                history.append(ACTIONS[order[choice - 1]])
            else:
                say(f"Pick a number between 1 and {len(order)}.")
        else:
            try:
                history.extend(parse_actions(entry))
            except ValueError as err:
                say(str(err))
    return history


def cmd_live(args) -> int:
    set_threads(args.threads)
    run_live(get_predictor(args.model, args.model_path), team=args.team, top_k=args.top_k)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="game-action-predictor",
                                     description="Frequent play sequences and next-action prediction.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that the tool runs")

    p = sub.add_parser("simulate", help="generate a season of raw play-by-play (data/play_by_play.csv)")
    p.add_argument("--games", type=int, default=80)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", default=str(storage.RAW_PATH))
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("prepare", help="clean raw play-by-play into possession sequences")
    p.add_argument("--input", default=str(storage.RAW_PATH))
    p.add_argument("--output", default=str(storage.SEQUENCES_PATH))
    p.set_defaults(func=cmd_prepare)

    p = sub.add_parser("mine", help="find frequent play sequences with thilimarketbasket")
    p.add_argument("--sequences", default=str(storage.SEQUENCES_PATH))
    p.add_argument("--min-support", type=float, default=0.01)
    p.add_argument("--max-size", type=int, default=3, help="largest itemset to mine")
    p.add_argument("--max-ngram", type=int, default=4, help="longest contiguous run added to each basket")
    p.add_argument("--top", type=int, default=10)
    p.set_defaults(func=cmd_mine)

    p = sub.add_parser("train", help="continue pretraining a language model on possession sequences")
    p.add_argument("--base", default=DEFAULT_BASE, help=f"Hugging Face causal LM (default {DEFAULT_BASE})")
    p.add_argument("--from-scratch", action="store_true", help="train a tiny randomly initialised GPT-2 instead")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--val-size", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (default 1, see README)")
    p.add_argument("--sequences", default=str(storage.SEQUENCES_PATH))
    p.add_argument("--out", default=str(storage.MODEL_DIR))
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="score the language model and baselines on held-out games")
    p.add_argument("--model", default=str(storage.MODEL_DIR), help="trained model folder")
    p.add_argument("--compare", nargs="*", default=[], metavar="NAME=DIR",
                   help="more trained models to compare, e.g. scratch=models/scratch")
    p.add_argument("--windows", nargs="*", type=int, default=[], metavar="K",
                   help="also evaluate the model seeing only the last K actions (sequence-length study)")
    p.add_argument("--cases", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--sequences", default=str(storage.SEQUENCES_PATH))
    p.set_defaults(func=cmd_evaluate)

    predictor_args = argparse.ArgumentParser(add_help=False)
    predictor_args.add_argument("--team", default="Comets", help="Comets, Titans, Wolves or Sharks")
    predictor_args.add_argument("-k", "--top-k", type=int, default=5)
    predictor_args.add_argument("--model", choices=["auto", "lm", "markov"], default="auto",
                                help="auto = the trained LM if it exists, else Markov-2")
    predictor_args.add_argument("--model-path", default=str(storage.MODEL_DIR))
    predictor_args.add_argument("--threads", type=int, default=1)

    p = sub.add_parser("predict", parents=[predictor_args], help="most likely next actions for a partial possession")
    p.add_argument("actions", nargs="?", default="", help='e.g. "inbound, bring up, pick and roll"')
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("live", parents=[predictor_args], help="interactive: follow a possession action by action")
    p.set_defaults(func=cmd_live)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from game-action-predictor-dashboard!")
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

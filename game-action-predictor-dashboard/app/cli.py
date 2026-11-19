"""Command-line interface for the game action predictor."""
from __future__ import annotations

import argparse

from app import storage
from app.core import clean_events, sequence_stats, to_sequences


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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from game-action-predictor-dashboard!")
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

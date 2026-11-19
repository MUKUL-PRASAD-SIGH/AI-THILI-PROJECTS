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
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "hello":
        print("Hello from game-action-predictor-dashboard!")
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

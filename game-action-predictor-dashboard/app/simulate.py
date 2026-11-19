"""A play-by-play simulator for basketball possessions, with planted team tendencies.

Real play-by-play feeds (NBA, FIBA) are licensed, so the project ships its own season, generated here.
Because we know the process that wrote it, we can check that the miner finds the patterns we planted and
compare the language model with the best accuracy any model could reach (`next_distribution` is the truth).

A possession is a chain of actions. The next action depends on
  * the previous action                      (base transition table),
  * the team's style                          (multipliers, see TEAM_STYLES),
  * the two previous actions                  (a planted second-order pattern: pick and roll -> pass -> three),
  * how long the possession has been going    (shot-clock pressure after 6 actions),
  * how the previous possession ended         (made basket -> inbound, miss -> defensive rebound, turnover -> steal).

The raw file is deliberately messy, like a real feed: action labels come in several spellings, timeouts and
substitutions are mixed in, and possessions are not marked. `core.clean_events` and `core.to_sequences` undo this.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

END = "<end>"

ACTIONS = [
    "inbound", "def_rebound", "steal",
    "bring_up", "fast_break",
    "pass", "dribble", "pick_and_roll", "handoff", "post_up", "drive", "kick_out", "off_rebound",
    "three_made", "three_missed", "jumper_made", "jumper_missed", "layup_made", "layup_missed", "dunk",
    "foul_drawn", "free_throw_made", "free_throw_missed", "turnover",
]
SHOTS = {"three_made", "three_missed", "jumper_made", "jumper_missed", "layup_made", "layup_missed", "dunk"}
MISSES = {"three_missed", "jumper_missed", "layup_missed"}

BASE = {
    "inbound": {"bring_up": 0.9, "pass": 0.1},
    "def_rebound": {"bring_up": 0.65, "fast_break": 0.35},
    "steal": {"fast_break": 0.8, "bring_up": 0.2},
    "bring_up": {"pass": 0.35, "pick_and_roll": 0.2, "dribble": 0.15, "handoff": 0.1, "post_up": 0.1,
                 "drive": 0.05, "turnover": 0.03, "three_made": 0.01, "three_missed": 0.01},
    "fast_break": {"layup_made": 0.35, "dunk": 0.2, "layup_missed": 0.1, "pass": 0.2, "foul_drawn": 0.1,
                   "turnover": 0.05},
    "pass": {"pass": 0.25, "pick_and_roll": 0.1, "drive": 0.15, "post_up": 0.1, "handoff": 0.08,
             "three_made": 0.07, "three_missed": 0.1, "jumper_made": 0.05, "jumper_missed": 0.05, "turnover": 0.05},
    "dribble": {"drive": 0.3, "pass": 0.3, "jumper_made": 0.1, "jumper_missed": 0.12, "pick_and_roll": 0.1,
                "turnover": 0.08},
    "pick_and_roll": {"drive": 0.35, "pass": 0.3, "jumper_made": 0.1, "jumper_missed": 0.1, "kick_out": 0.1,
                      "turnover": 0.05},
    "handoff": {"drive": 0.3, "three_made": 0.15, "three_missed": 0.2, "pass": 0.25, "pick_and_roll": 0.1},
    "post_up": {"jumper_made": 0.25, "jumper_missed": 0.25, "foul_drawn": 0.2, "kick_out": 0.2, "turnover": 0.1},
    "drive": {"layup_made": 0.3, "layup_missed": 0.2, "dunk": 0.1, "foul_drawn": 0.15, "kick_out": 0.2,
              "turnover": 0.05},
    "kick_out": {"three_made": 0.3, "three_missed": 0.4, "pass": 0.2, "drive": 0.1},
    "off_rebound": {"layup_made": 0.3, "layup_missed": 0.15, "kick_out": 0.25, "pass": 0.3},
    "foul_drawn": {"free_throw_made": 0.75, "free_throw_missed": 0.25},
}
for _miss in MISSES:
    BASE[_miss] = {"off_rebound": 0.27, END: 0.73}

# Each team multiplies the weight of some actions wherever they can happen.
TEAM_STYLES = {
    "Comets": {"pick_and_roll": 3.0, "drive": 1.5},                      # pick-and-roll offence
    "Titans": {"post_up": 3.5, "foul_drawn": 1.5},                       # inside, post play
    "Wolves": {"fast_break": 2.5, "dunk": 1.5, "turnover": 1.5},         # pace and transition
    "Sharks": {"pass": 1.5, "kick_out": 2.0, "three_made": 1.6, "three_missed": 1.6},  # ball movement, threes
}
TEAMS = list(TEAM_STYLES)

# Several spellings of the same action, as different scorers / feeds write them.
RAW_LABELS = {
    "inbound": ["Inbound", "inbound pass"], "def_rebound": ["Defensive Rebound", "DREB"],
    "steal": ["Steal", "STL"], "bring_up": ["Brings ball up", "Bring Up"], "fast_break": ["Fast Break", "Transition"],
    "pass": ["Pass", "PASS"], "dribble": ["Dribble", "Iso dribble"], "pick_and_roll": ["Pick & Roll", "PnR"],
    "handoff": ["Hand-off", "DHO"], "post_up": ["Post Up", "Post-up"], "drive": ["Drive", "Drive to basket"],
    "kick_out": ["Kick Out", "Kick-out pass"], "off_rebound": ["Offensive Rebound", "OREB"],
    "three_made": ["3PT Made", "Three Pointer (made)"], "three_missed": ["3PT Missed", "Three Pointer (missed)"],
    "jumper_made": ["Jump Shot Made", "2PT Jumper (made)"], "jumper_missed": ["Jump Shot Missed", "2PT Jumper (missed)"],
    "layup_made": ["Layup Made", "Layup (made)"], "layup_missed": ["Layup Missed", "Layup (missed)"],
    "dunk": ["Dunk", "Slam Dunk"], "foul_drawn": ["Shooting Foul", "Foul drawn"],
    "free_throw_made": ["Free Throw Made", "FT made"], "free_throw_missed": ["Free Throw Missed", "FT missed"],
    "turnover": ["Turnover", "TOV"],
}
NOISE_EVENTS = ["Timeout", "Substitution", "Jump Ball Violation", "Review"]


def next_distribution(team: str, history: list[str], start: str | None = None) -> dict[str, float]:
    """The true probability of each next action (and of END) given the team and the actions so far.

    `start` is how the possession begins; it only matters when `history` is empty.
    """
    if not history:
        return {start or "inbound": 1.0}
    last = history[-1]
    if last in {"three_made", "jumper_made", "layup_made", "dunk", "turnover"}:
        return {END: 1.0}
    if last in {"free_throw_made", "free_throw_missed"}:
        # Two free throws after a shooting foul; after the second, a miss can be rebounded.
        first_of_two = len(history) >= 2 and history[-2] == "foul_drawn"
        if first_of_two:
            return {"free_throw_made": 0.75, "free_throw_missed": 0.25}
        return {"off_rebound": 0.15, END: 0.85} if last == "free_throw_missed" else {END: 1.0}

    weights = dict(BASE[last])
    style = TEAM_STYLES[team]
    for action in weights:
        weights[action] *= style.get(action, 1.0)
    if history[-2:] == ["pick_and_roll", "pass"]:          # planted second-order pattern: roll, pass, three
        for action in ("three_made", "three_missed"):
            weights[action] = weights.get(action, 0.02) * 3.0
    if len(history) >= 7:                                  # shot clock running down
        for action in list(weights):
            if action in SHOTS or action == "turnover":
                weights[action] *= 2.5
            elif action == "pass":
                weights[action] *= 0.5
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


def simulate_possession(team: str, start: str, rng: np.random.Generator, max_len: int = 25) -> list[str]:
    history: list[str] = []
    while len(history) < max_len:
        dist = next_distribution(team, history, start)
        actions = list(dist)
        action = actions[rng.choice(len(actions), p=np.array([dist[a] for a in actions]))]
        if action == END:
            break
        history.append(action)
    return history


def _next_start(previous: list[str]) -> str:
    last = previous[-1] if previous else "inbound"
    if last == "turnover":
        return "steal"
    if last in MISSES or last == "free_throw_missed":
        return "def_rebound"
    return "inbound"


def simulate_season(n_games: int = 80, possessions_per_team: int = 90, seed: int = 0,
                    noise: bool = True) -> pd.DataFrame:
    """A season of messy raw play-by-play, one row per event."""
    rng = np.random.default_rng(seed)
    rows = []
    for game in range(1, n_games + 1):
        home, away = rng.choice(TEAMS, size=2, replace=False)
        offence, previous = (home, away), []
        seconds = 0.0
        event = 0
        for number in range(2 * possessions_per_team):
            team = offence[number % 2]
            actions = simulate_possession(team, _next_start(previous), rng)
            previous = actions
            for action in actions:
                seconds += float(rng.uniform(1.0, 4.0))
                period = min(4, int(seconds // 720) + 1)
                left = max(0.0, 720 * period - seconds)
                event += 1
                label = RAW_LABELS[action][int(rng.integers(len(RAW_LABELS[action])))] if noise else action
                rows.append({"game_id": f"G{game:03d}", "event_id": event, "period": period,
                             "clock": f"{int(left // 60):02d}:{int(left % 60):02d}", "team": team,
                             "opponent": away if team == home else home, "event": label})
                if noise and rng.random() < 0.015:
                    event += 1
                    rows.append({"game_id": f"G{game:03d}", "event_id": event, "period": period,
                                 "clock": rows[-1]["clock"], "team": team if rng.random() < 0.5 else "",
                                 "opponent": "", "event": NOISE_EVENTS[int(rng.integers(len(NOISE_EVENTS)))]})
    return pd.DataFrame(rows)

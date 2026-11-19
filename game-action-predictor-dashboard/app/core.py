"""Core logic: play-by-play -> event sequences -> baskets, frequent-sequence mining, baselines, evaluation."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.simulate import ACTIONS, RAW_LABELS

# ---------------------------------------------------------------------------
# Milestone 1: play-by-play -> event sequences -> baskets
# ---------------------------------------------------------------------------


def _key(label: str) -> str:
    """Spelling-insensitive key: lowercase letters and digits only ("Pick & Roll" -> "pickroll")."""
    return re.sub(r"[^a-z0-9]", "", str(label).lower())


# Every known spelling (and the canonical name itself) maps to one canonical action.
ALIASES: dict[str, str] = {}
for _action, _labels in RAW_LABELS.items():
    for _label in [_action, _action.replace("_", " "), *_labels]:
        ALIASES[_key(_label)] = _action


def normalize_action(label: str) -> str | None:
    """Canonical action name for a raw event label, or None if it is not a play (timeout, substitution...)."""
    return ALIASES.get(_key(label))


def clean_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalise labels, drop non-play rows (timeouts, substitutions, unknown labels, missing team),
    and put events back in game order."""
    df = raw.copy()
    df["action"] = df["event"].map(normalize_action)
    df = df[df["action"].notna() & df["team"].fillna("").astype(str).str.len().gt(0)]
    return df.sort_values(["game_id", "event_id"]).reset_index(drop=True)


@dataclass
class Possession:
    game_id: str
    team: str
    number: int                     # possession number within the game
    actions: list[str] = field(default_factory=list)


def to_sequences(events: pd.DataFrame) -> list[Possession]:
    """Group cleaned events into possessions: a possession is a run of consecutive events by one team."""
    possessions: list[Possession] = []
    for game_id, game in events.groupby("game_id", sort=True):
        current: Possession | None = None
        number = 0
        for team, action in zip(game["team"], game["action"]):
            if current is None or team != current.team:
                number += 1
                current = Possession(str(game_id), str(team), number)
                possessions.append(current)
            current.actions.append(action)
    return possessions


TRANSITION = ">"


def to_basket(actions: list[str], team: str | None = None, transitions: bool = True,
              max_ngram: int = 1) -> list[str]:
    """One possession as a basket (a set of items) for the market-basket miner.

    A basket has no order, but a play does. So, besides the actions themselves, the basket holds
      * every ordered transition "a>b" between consecutive actions (with `transitions`), and
      * every contiguous run of up to `max_ngram` actions, written "a>b>c" (with `max_ngram` > 2),
    which lets an itemset miner find frequent *sequences*, not just actions that co-occur.
    A "team=X" item lets the miner relate patterns to teams.
    """
    items = set(actions)
    if transitions:
        items.update(f"{a}{TRANSITION}{b}" for a, b in zip(actions, actions[1:]))
    for n in range(3, max_ngram + 1):
        items.update(TRANSITION.join(actions[i:i + n]) for i in range(len(actions) - n + 1))
    if team:
        items.add(f"team={team}")
    return sorted(items)


def to_baskets(possessions: list[Possession], **kwargs) -> list[list[str]]:
    return [to_basket(p.actions, p.team, **kwargs) for p in possessions]


def split_by_game(possessions: list[Possession], test_share: float = 0.15) -> tuple[list[Possession], list[Possession]]:
    """Hold out whole games (the last `test_share` of them), so no test possession leaks into training."""
    games = sorted({p.game_id for p in possessions})
    cut = int(round(len(games) * (1 - test_share)))
    test_games = set(games[cut:])
    return [p for p in possessions if p.game_id not in test_games], [p for p in possessions if p.game_id in test_games]


def sequence_stats(possessions: list[Possession]) -> dict:
    lengths = [len(p.actions) for p in possessions]
    counts = Counter(a for p in possessions for a in p.actions)
    return {"possessions": len(possessions), "games": len({p.game_id for p in possessions}),
            "events": int(sum(lengths)), "mean_length": float(np.mean(lengths)), "max_length": int(max(lengths)),
            "actions": len(counts), "most_common": counts.most_common(5)}


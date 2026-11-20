"""How well is the next action predicted? Models, baselines and metrics. (Milestone 4.)

Every predictor maps a list of contexts (team, actions so far) to a probability for each action in ACTIONS.
The test cases come from held-out games: in every test possession, every point after the first action is a
case, "given these actions, what comes next?". Metrics follow thilimarketbasket.evaluate: the rank of the
true next action gives top-1 accuracy (hits@1), hits@3, hits@5 and MRR; we add log-loss, which also rewards
being confident when right and punishes being confident when wrong.

Baselines, weakest to strongest:
  popularity   - the league-wide frequency of each action; never reads the possession
  item-to-item - thilimarketbasket.recommend.ItemToItem: what co-occurs with the actions seen (no order)
  markov-k     - counts of what followed the last k actions for this team, backing off to shorter histories
  oracle       - the simulator's true probabilities: the best any model can do on this data
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Sequence

import numpy as np

from app.core import Possession
from app.simulate import ACTIONS, END, next_distribution

INDEX = {a: i for i, a in enumerate(ACTIONS)}
Context = tuple[str, Sequence[str]]


def make_cases(possessions: Sequence[Possession], n_cases: int | None = 1000, seed: int = 0,
               min_history: int = 1) -> list[dict]:
    """(team, history, target) for every position after `min_history` actions; a random `n_cases` of them."""
    cases = [{"team": p.team, "history": p.actions[:t], "target": p.actions[t]}
             for p in possessions for t in range(min_history, len(p.actions))]
    if n_cases is not None and n_cases < len(cases):
        rng = np.random.default_rng(seed)
        cases = [cases[i] for i in sorted(rng.choice(len(cases), size=n_cases, replace=False))]
    return cases


# ---------------------------------------------------------------------------
# predictors
# ---------------------------------------------------------------------------

class Popularity:
    name = "popularity"

    def __init__(self, train: Sequence[Possession]):
        counts = Counter(a for p in train for a in p.actions)
        self.probs = np.array([counts[a] + 1.0 for a in ACTIONS])
        self.probs /= self.probs.sum()

    def predict_proba(self, contexts: Sequence[Context]) -> np.ndarray:
        return np.tile(self.probs, (len(contexts), 1))


class Markov:
    """P(next | team, last k actions), with Dirichlet back-off to shorter histories:

        P_k(w) = (count_k(w) + beta * P_{k-1}(w)) / (count_k + beta),   P_0 = this team's action frequencies
    """

    def __init__(self, train: Sequence[Possession], order: int = 2, beta: float = 1.0):
        self.order, self.beta, self.name = order, beta, f"markov-{order}"
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        for p in train:
            for t in range(len(p.actions)):
                for k in range(0, order + 1):
                    if k <= t:
                        self.counts[(p.team, *p.actions[t - k:t])][p.actions[t]] += 1

    def _dist(self, team: str, history: Sequence[str]) -> np.ndarray:
        probs = np.full(len(ACTIONS), 1.0 / len(ACTIONS))
        for k in range(0, min(self.order, len(history)) + 1):
            counts = self.counts.get((team, *history[len(history) - k:]))
            if not counts:
                break
            total = sum(counts.values())
            vec = np.array([counts.get(a, 0) for a in ACTIONS], dtype=float)
            probs = (vec + self.beta * probs) / (total + self.beta)
        return probs

    def predict_proba(self, contexts: Sequence[Context]) -> np.ndarray:
        return np.array([self._dist(team, history) for team, history in contexts])


class ItemToItemBaseline:
    """thilimarketbasket's counting recommender, on possessions-as-baskets: it knows which actions appear
    together, but not in what order."""
    name = "item-to-item"

    def __init__(self, train: Sequence[Possession], eps: float = 1e-3):
        from thilimarketbasket.recommend import ItemToItem

        self.model = ItemToItem([p.actions for p in train], ACTIONS)
        # Every item "co-occurs" with itself in 100% of its baskets, so the raw scores always pick an action
        # already seen. thilimarketbasket excludes visible items instead; possessions repeat actions
        # (pass, pass), so we keep them as candidates and only drop the self term.
        np.fill_diagonal(self.model.together, 0.0)
        self.eps = eps

    def predict_proba(self, contexts: Sequence[Context]) -> np.ndarray:
        out = []
        for _team, history in contexts:
            scores = self.model.score(list(dict.fromkeys(history))) + self.eps
            out.append(scores / scores.sum())
        return np.array(out)


class Oracle:
    """The simulator's true next-action probabilities (END removed and the rest renormalised)."""
    name = "oracle (true process)"

    def predict_proba(self, contexts: Sequence[Context]) -> np.ndarray:
        out = np.full((len(contexts), len(ACTIONS)), 1e-9)
        for row, (team, history) in enumerate(contexts):
            for action, p in next_distribution(team, list(history)).items():
                if action != END:
                    out[row, INDEX[action]] += p
        return out / out.sum(axis=1, keepdims=True)


class LanguageModel:
    """Adapter so the trained ActionLM looks like the other predictors."""

    def __init__(self, lm, name: str = "language model", window: int | None = None):
        self.lm, self.name, self.window = lm, name, window

    def predict_proba(self, contexts: Sequence[Context]) -> np.ndarray:
        if self.window is not None:
            contexts = [(team, list(history)[-self.window:]) for team, history in contexts]
        return self.lm.predict_proba(contexts)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def score(probs: np.ndarray, cases: Sequence[dict]) -> dict:
    """hits@1/3/5, MRR, median rank (thilimarketbasket.evaluate) plus log-loss, from predicted probabilities."""
    from thilimarketbasket.evaluate import rank_of, summarise

    targets = [INDEX[c["target"]] for c in cases]
    ranks = [rank_of(probs[i], t) for i, t in enumerate(targets)]
    out = summarise(ranks, ks=(1, 3, 5))
    out["accuracy"] = out["hits@1"]
    out["log_loss"] = float(-np.mean([np.log(max(probs[i, t], 1e-12)) for i, t in enumerate(targets)]))
    out["ranks"] = ranks
    return out


def by_history_length(ranks: Sequence[int], cases: Sequence[dict], buckets=(1, 2, 3, 4, 5, 6)) -> dict[str, float]:
    """Top-1 accuracy split by how many actions of the possession were seen (the last bucket is "or more")."""
    out = {}
    for i, low in enumerate(buckets):
        last = i == len(buckets) - 1
        chosen = [r for r, c in zip(ranks, cases)
                  if len(c["history"]) == low or (last and len(c["history"]) >= low)]
        out[f"{low}+" if last else str(low)] = float(np.mean([r == 1 for r in chosen])) if chosen else float("nan")
    return out


def evaluate_all(predictors: Sequence, cases: Sequence[dict]) -> dict[str, dict]:
    contexts = [(c["team"], c["history"]) for c in cases]
    results = {}
    for predictor in predictors:
        metrics = score(predictor.predict_proba(contexts), cases)
        metrics["by_length"] = by_history_length(metrics["ranks"], cases)
        results[predictor.name] = metrics
    return results


def results_table(results: dict[str, dict]) -> str:
    lines = ["| model | top-1 accuracy | hits@3 | hits@5 | MRR | log-loss |", "|---|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['accuracy']:.1%} | {m['hits@3']:.1%} | {m['hits@5']:.1%} | "
                     f"{m['mrr']:.3f} | {m['log_loss']:.3f} |")
    return "\n".join(lines)


def length_table(results: dict[str, dict]) -> str:
    keys = list(next(iter(results.values()))["by_length"])
    lines = ["| model | " + " | ".join(f"{k} seen" for k in keys) + " |", "|---|" + "---|" * len(keys)]
    for name, m in results.items():
        lines.append(f"| {name} | " + " | ".join(f"{m['by_length'][k]:.1%}" for k in keys) + " |")
    return "\n".join(lines)

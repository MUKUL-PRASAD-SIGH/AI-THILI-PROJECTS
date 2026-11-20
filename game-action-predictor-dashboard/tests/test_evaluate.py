import numpy as np
import pytest

from app import storage
from app.core import Possession, split_by_game
from app.evaluate import (ItemToItemBaseline, Markov, Oracle, Popularity, by_history_length, evaluate_all,
                          make_cases, score)
from app.simulate import ACTIONS


@pytest.fixture(scope="module")
def split():
    return split_by_game(storage.read_sequences())


@pytest.fixture(scope="module")
def results(split):
    train, test = split
    cases = make_cases(test, 2000, seed=0)
    return evaluate_all([Popularity(train), ItemToItemBaseline(train), Markov(train, 1), Markov(train, 2), Oracle()],
                        cases)


def test_make_cases_covers_every_position():
    poss = [Possession("G1", "Comets", 1, ["inbound", "bring_up", "pass", "three_made"])]
    cases = make_cases(poss, None)
    assert [c["target"] for c in cases] == ["bring_up", "pass", "three_made"]
    assert [len(c["history"]) for c in cases] == [1, 2, 3]


def test_score_metrics_on_a_known_example():
    probs = np.full((2, len(ACTIONS)), 0.01)
    probs[0, ACTIONS.index("pass")] = 0.9           # right, ranked 1
    probs[1, ACTIONS.index("pass")] = 0.9           # wrong; target "drive" ranked 2
    probs[1, ACTIONS.index("drive")] = 0.5
    probs /= probs.sum(axis=1, keepdims=True)
    m = score(probs, [{"target": "pass"}, {"target": "drive"}])
    assert m["accuracy"] == 0.5 and m["hits@3"] == 1.0 and m["mrr"] == pytest.approx(0.75)


@pytest.mark.parametrize("name", ["popularity", "item-to-item", "markov-1", "markov-2", "oracle (true process)"])
def test_predictors_return_distributions(split, name):
    train, _ = split
    predictor = {"popularity": Popularity(train), "item-to-item": ItemToItemBaseline(train),
                 "markov-1": Markov(train, 1), "markov-2": Markov(train, 2), "oracle (true process)": Oracle()}[name]
    probs = predictor.predict_proba([("Comets", ["inbound"]), ("Titans", ["inbound", "bring_up", "post_up"])])
    assert probs.shape == (2, len(ACTIONS)) and np.allclose(probs.sum(axis=1), 1.0)


def test_models_beat_popularity_and_approach_the_oracle(results):
    acc = {k: v["accuracy"] for k, v in results.items()}
    assert acc["markov-1"] > acc["popularity"] + 0.2
    assert acc["item-to-item"] > 0.05                         # the self-co-occurrence fix works
    assert abs(acc["markov-2"] - acc["oracle (true process)"]) < 0.03
    # the oracle is the best-calibrated by construction
    assert results["oracle (true process)"]["log_loss"] <= min(m["log_loss"] for m in results.values()) + 0.02


def test_by_history_length_buckets():
    cases = [{"history": ["a"] * n} for n in (1, 1, 2, 7)]
    out = by_history_length([1, 2, 1, 1], cases)
    assert out["1"] == 0.5 and out["2"] == 1.0 and out["6+"] == 1.0

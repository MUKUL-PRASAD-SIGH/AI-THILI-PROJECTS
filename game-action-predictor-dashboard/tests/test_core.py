import math

import pytest

from app import storage
from app.core import (clean_events, normalize_action, split_by_game, to_basket, to_sequences)
from app.simulate import ACTIONS, END, TEAMS, next_distribution, simulate_season


@pytest.fixture(scope="module")
def possessions():
    return storage.read_sequences()


# --- Milestone 1: play-by-play -> sequences -> baskets ----------------------

def test_simulator_distributions_are_valid():
    for team in TEAMS:
        for history in ([], ["inbound"], ["inbound", "bring_up", "pick_and_roll", "pass"], ["foul_drawn",
                        "free_throw_made"], ["pass"] * 8):
            dist = next_distribution(team, history)
            assert math.isclose(sum(dist.values()), 1.0)
            assert set(dist) <= set(ACTIONS) | {END}


def test_normalize_action_handles_spellings_and_noise():
    assert normalize_action("Pick & Roll") == normalize_action("PnR") == "pick_and_roll"
    assert normalize_action("3PT Made") == normalize_action("three made") == "three_made"
    assert normalize_action("Timeout") is None and normalize_action("Substitution") is None


def test_clean_and_group_recovers_possessions():
    raw = simulate_season(n_games=3, possessions_per_team=20, seed=1)
    events = clean_events(raw)
    assert set(events["action"]) <= set(ACTIONS)
    assert len(events) < len(raw)                      # noise rows were dropped
    poss = to_sequences(events)
    assert len(poss) == 3 * 40                         # possessions alternate, so a team change marks one
    assert all(p.actions for p in poss)


def test_clean_events_sorts_shuffled_rows():
    raw = simulate_season(n_games=1, possessions_per_team=5, seed=2)
    shuffled = raw.sample(frac=1.0, random_state=0)
    assert [p.actions for p in to_sequences(clean_events(shuffled))] == \
           [p.actions for p in to_sequences(clean_events(raw))]


def test_to_basket_has_actions_transitions_and_team():
    basket = to_basket(["inbound", "bring_up", "pass", "pass"], "Comets", max_ngram=3)
    assert {"inbound", "bring_up", "pass", "team=Comets"} <= set(basket)
    assert {"inbound>bring_up", "bring_up>pass", "pass>pass", "inbound>bring_up>pass"} <= set(basket)
    assert len(basket) == len(set(basket))


def test_prepared_data_and_game_split(possessions):
    assert len(possessions) > 10_000
    train, test = split_by_game(possessions)
    assert not {p.game_id for p in train} & {p.game_id for p in test}
    assert 0.1 < len(test) / len(possessions) < 0.2

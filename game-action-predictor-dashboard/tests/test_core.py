import math

import pytest

from app import storage
from app.core import (clean_events, mine_patterns, normalize_action, split_by_game, team_lift, to_basket,
                      to_sequences)
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


# --- Milestone 2: frequent sequences with thilimarketbasket ------------------

@pytest.fixture(scope="module")
def patterns(possessions):
    return mine_patterns(possessions)


def test_mining_finds_frequent_sequences(patterns):
    top = [s for s, _ in patterns["sequences"][:5]]
    assert top[0] == "inbound>bring_up"                     # how most possessions start
    assert all(">" in s for s in top)
    supports = [sup for _, sup in patterns["sequences"]]
    assert supports == sorted(supports, reverse=True) and min(supports) >= 0.01


@pytest.mark.parametrize("team,play", [("Comets", "pick_and_roll>drive"), ("Titans", "post_up"),
                                       ("Wolves", "fast_break"), ("Sharks", "kick_out")])
def test_mining_recovers_planted_team_styles(patterns, team, play):
    assert team_lift(patterns, team, play) > 1.15
    others = [team_lift(patterns, t, play) for t in TEAMS if t != team]
    assert team_lift(patterns, team, play) > max(x for x in others if x == x)  # highest of all teams (nan-safe)


def test_outcome_rules_are_play_to_outcome(patterns):
    assert patterns["outcomes"]
    for play, outcome, support, confidence, lift in patterns["outcomes"]:
        assert ">" in play and outcome not in play.split(">") and confidence >= 0.3
    kick = [r for r in patterns["outcomes"] if r[0] == "drive>kick_out"]
    assert {r[1] for r in kick} >= {"three_missed", "three_made"}

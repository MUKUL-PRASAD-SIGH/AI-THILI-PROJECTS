"""Language-model tests. They use a tiny randomly initialised model (no checkpoint download, only the
tokenizer), so they run in CI; the real continued-pretraining run is `python -m app.cli train`."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app import storage  # noqa: E402
from app.core import split_by_game  # noqa: E402
from app.lm import ActionLM, possession_text, prefix_text  # noqa: E402
from app.simulate import ACTIONS  # noqa: E402

torch.set_num_threads(1)


@pytest.fixture(scope="module")
def data():
    train, test = split_by_game(storage.read_sequences())
    return train[:400], test[:60]


@pytest.fixture(scope="module")
def tiny():
    return ActionLM.tiny(seed=0, n_layer=1, n_embd=64)


def test_texts_line_up():
    actions = ["inbound", "bring_up", "pick_and_roll"]
    full = possession_text("Comets", actions)
    assert full == "Possession (Comets): inbound; bring up; pick and roll"
    for k in range(len(actions)):
        assert full.startswith(prefix_text("Comets", actions[:k]))
    assert prefix_text("Comets", []) == "Possession (Comets):"


def test_every_action_is_a_candidate(tiny):
    assert tiny.actions == ACTIONS and len(tiny.candidate_ids) == len(ACTIONS)
    assert len({tuple(c) for c in tiny.candidate_ids}) == len(ACTIONS)


def test_predict_proba_shape_and_normalisation(tiny):
    probs = tiny.predict_proba([("Comets", []), ("Sharks", ["inbound", "bring_up"])])
    assert probs.shape == (2, len(ACTIONS))
    assert np.allclose(probs.sum(axis=1), 1.0) and (probs > 0).all()


def test_batching_does_not_change_scores(tiny):
    ctx = [("Titans", ["inbound", "bring_up", "post_up"]), ("Wolves", ["steal"])]
    assert np.allclose(tiny.score_batch(ctx, batch_rows=7), tiny.score_batch(ctx, batch_rows=500), atol=1e-4)


def test_continued_pretraining_lowers_validation_loss_and_learns(data):
    train, test = data
    lm = ActionLM.tiny(seed=1, n_layer=1, n_embd=64)
    bring_up = ACTIONS.index("bring_up")
    before = lm.predict_proba([("Comets", ["inbound"])])[0][bring_up]
    history = lm.pretrain(train, test[:30], steps=40, batch_size=16, learning_rate=3e-3, warmup=5, eval_every=20,
                          log=lambda s: None)
    assert history[-1]["val_loss"] < 0.5 * history[0]["val_loss"]
    # after "inbound" the next action is "bring_up" 90% of the time: the model should have moved towards it
    after = lm.predict_proba([("Comets", ["inbound"])])[0][bring_up]
    assert after > 100 * before


def test_save_and_load_round_trip(tiny, tmp_path):
    out = tiny.save(tmp_path / "m", history=[{"step": 0}], meta={"base": "tiny"})
    again = ActionLM.load(str(out))
    ctx = [("Sharks", ["inbound"])]
    assert np.allclose(again.score_batch(ctx), tiny.score_batch(ctx), atol=1e-5)
    assert (out / "training.json").exists()

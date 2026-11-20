import json

from app.core import LABELS, Example
from app.llm import Sorter, build_model_and_tokenizer
from app.train import Collator, RequestDataset, train


def test_dataset_masks_everything_but_the_category(tiny_base):
    _, tok = build_model_and_tokenizer(tiny_base)
    ds = RequestDataset([Example("Is the hall free?", "venue_inquiry")], tok)
    item = ds[0]
    supervised = [t for t in item["labels"] if t != -100]
    assert supervised == [tok.convert_tokens_to_ids("<|venue_inquiry|>"), tok.eos_token_id]
    batch = Collator(tok.pad_token_id)([item, ds[0] | {"input_ids": item["input_ids"][:3],
                                                         "labels": item["labels"][:3]}])
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    assert batch["attention_mask"][1].sum() == 3


def test_trainer_fine_tunes_and_saves(tiny_trained):
    out, metrics = tiny_trained
    assert metrics["after"]["accuracy"] > metrics["before"]["accuracy"]
    assert metrics["after"]["accuracy"] >= 0.75
    assert any("eval_accuracy" in h for h in metrics["history"])
    saved = json.loads((out / "metrics.json").read_text())
    assert set(saved["after"]["per_class"]) == set(LABELS)
    assert not (out / "_trainer").exists()
    pred = Sorter(out, device="cpu").classify("note 99, please help with photos")
    assert pred.label == "photo_video"

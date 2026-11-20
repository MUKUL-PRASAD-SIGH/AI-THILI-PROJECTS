"""Measure whether generated answers are accurate and relevant.

Quantitative, per answer (all computed against the approved answer for that FAQ):

* ``fact_recall``   share of the FAQ's key facts (fees, deadlines, phone numbers,
                    places) that appear in the generated answer. This is the main
                    accuracy number: a resident acting on the answer needs these.
* ``all_facts``     1 if every key fact is present.
* ``unsupported``   1 if the answer states a number (fee, date, phone, count)
                    that is not in the approved answer, i.e. a likely hallucination.
* ``relevant``      1 if, out of all 60 approved answers, the generated answer is
                    closest (token F1) to the right one: the bot answered the
                    question that was asked, not a neighbouring FAQ.
* ``token_f1``, ``rouge_l``, ``exact``  standard overlap scores against the reference.

Qualitative: ``write_report`` puts a side-by-side table of questions, approved
answers and each model's answer into ``reports/evaluation.md`` for a person to read.

Systems compared: the pre-trained model with the same prompt (no adaptation), a
word-overlap lookup baseline that returns the stored answer of the most similar
training question, and the LoRA fine-tuned model.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from dataclasses import dataclass, field

from app import core

_PUNCT = str.maketrans("", "", string.punctuation.replace("$", "").replace("%", ""))
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?(?::\d\d)?(?:-\d+)?")
STOPWORDS = frozenset("a an the and or of to in on at for by is are be can do does i my me you your "
                      "it what how when where which who why with from this that there".split())


def normalize(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCT).split())


def tokens(text: str) -> list[str]:
    return normalize(text).split()


def token_f1(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return float(p == r)
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred: str, ref: str) -> float:
    p, r = tokens(pred), tokens(ref)
    if not p or not r:
        return float(p == r)
    lcs = _lcs(p, r)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(r)
    return 2 * precision * recall / (precision + recall)


def fact_hits(pred: str, facts: tuple[str, ...] | list[str]) -> list[bool]:
    norm = normalize(pred)
    return [normalize(f) in norm for f in facts]


def numbers(text: str) -> set[str]:
    return {m.group().replace(",", "").rstrip(".") for m in _NUMBER.finditer(text)}


def unsupported_numbers(pred: str, ref: str) -> set[str]:
    """Numbers in the prediction that the approved answer does not contain."""
    return numbers(pred) - numbers(ref)


def closest_answer(pred: str, faqs: list[core.FAQ]) -> core.FAQ:
    return max(faqs, key=lambda f: token_f1(pred, f.answer))


def score_answer(pred: str, example: core.Example, faqs: list[core.FAQ]) -> dict:
    hits = fact_hits(pred, example.facts)
    extra = unsupported_numbers(pred, example.answer)
    return {
        "faq_id": example.faq_id,
        "department": example.department,
        "question": example.question,
        "reference": example.answer,
        "prediction": pred,
        "fact_recall": sum(hits) / len(hits) if hits else 1.0,
        "all_facts": float(all(hits)),
        "missing_facts": [f for f, h in zip(example.facts, hits) if not h],
        "unsupported": float(bool(extra)),
        "unsupported_numbers": sorted(extra),
        "relevant": float(closest_answer(pred, faqs).id == example.faq_id),
        "token_f1": token_f1(pred, example.answer),
        "rouge_l": rouge_l(pred, example.answer),
        "exact": float(normalize(pred) == normalize(example.answer)),
    }


METRICS = ("fact_recall", "all_facts", "relevant", "unsupported", "token_f1", "rouge_l", "exact")


def summarize(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {m: 0.0 for m in METRICS}
    return {m: sum(r[m] for r in rows) / len(rows) for m in METRICS}


def by_department(rows: list[dict]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r["department"], []).append(r)
    return {d: summarize(rs) | {"n": len(rs)} for d, rs in sorted(groups.items())}


# --- lookup baseline ---------------------------------------------------------

def content_words(text: str) -> set[str]:
    return {t for t in tokens(text) if t not in STOPWORDS}


@dataclass
class LookupBaseline:
    """Return the stored answer of the training question with the most word overlap (Jaccard)."""

    examples: list[core.Example] = field(default_factory=list)

    def match(self, question: str) -> core.Example:
        q = content_words(question)

        def jaccard(e: core.Example) -> float:
            w = content_words(e.question)
            return len(q & w) / len(q | w) if q | w else 0.0

        return max(self.examples, key=jaccard)

    def answer(self, question: str) -> str:
        return self.match(question).answer


# --- running a full evaluation ----------------------------------------------

def evaluate_predictions(name: str, preds: list[str], examples: list[core.Example],
                         faqs: list[core.FAQ]) -> dict:
    rows = [score_answer(p, e, faqs) for p, e in zip(preds, examples)]
    return {"system": name, "summary": summarize(rows), "by_department": by_department(rows), "rows": rows}


def evaluate_model(name: str, model, tokenizer, examples: list[core.Example], faqs: list[core.FAQ]) -> dict:
    from app import llm

    preds = llm.generate_answers(model, tokenizer, [e.question for e in examples])
    return evaluate_predictions(name, preds, examples, faqs)


def _fmt(x: float) -> str:
    return f"{100 * x:.1f}%"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ") or "*(empty)*"


def write_report(results: dict, path, n_examples: int = 12) -> str:
    """Markdown report: summary tables per split, per department, then side-by-side samples."""
    lines = ["# Evaluation: Public FAQ Answer Bot", ""]
    meta = results.get("meta", {})
    if meta:
        lines += [f"- Base model: `{meta.get('base_model')}`",
                  f"- Adapter: r={meta.get('r')}, alpha={meta.get('alpha')}, "
                  f"targets {', '.join(meta.get('targets', []))}",
                  f"- FAQs: {meta.get('n_faqs')} across {meta.get('n_departments')} departments; "
                  f"decoding is greedy", ""]
    lines += ["Metrics are explained in `app/evaluate.py`. *Held-out* questions are paraphrases never "
              "seen in training; *seen* questions are the canonical wordings used in training. "
              "For **unsupported numbers**, lower is better.", ""]
    for split, systems in results["splits"].items():
        lines += [f"## {split}", "", "| System | Fact recall | All facts | Relevant | Unsupported numbers "
                  "| Token F1 | ROUGE-L | Exact |", "|---|---|---|---|---|---|---|---|"]
        for res in systems:
            s = res["summary"]
            lines.append(f"| {res['system']} | " + " | ".join(_fmt(s[m]) for m in METRICS) + " |")
        lines.append("")

    held = results["splits"].get("Held-out paraphrases", [])
    lora_res = next((r for r in held if r["system"].startswith("LoRA")), None)
    if lora_res:
        lines += ["## Held-out accuracy by department (LoRA model)", "",
                  "| Department | n | Fact recall | All facts | Relevant |", "|---|---|---|---|---|"]
        for dept, s in lora_res["by_department"].items():
            lines.append(f"| {dept} | {s['n']} | {_fmt(s['fact_recall'])} | {_fmt(s['all_facts'])} "
                         f"| {_fmt(s['relevant'])} |")
        lines.append("")

    if held:
        lines += ["## Sample answers to held-out questions", ""]
        by_sys = {r["system"]: r["rows"] for r in held}
        names = list(by_sys)
        step = max(1, len(held[0]["rows"]) // n_examples)
        for i in range(0, len(held[0]["rows"]), step)[:n_examples]:
            row = held[0]["rows"][i]
            lines += [f"**Q ({row['department']}):** {row['question']}", "",
                      f"- *Approved:* {row['reference']}"]
            for n in names:
                r = by_sys[n][i]
                flag = "all facts" if r["all_facts"] else f"missing {', '.join(r['missing_facts'])}"
                lines.append(f"- *{n}:* {_cell(r['prediction'])} — {flag}")
            lines.append("")

    if lora_res:
        misses = [r for r in lora_res["rows"] if not r["all_facts"] or r["unsupported"]]
        lines += ["## Where the LoRA model went wrong (held-out)", ""]
        if not misses:
            lines.append("No misses: every held-out answer contained all key facts and no unsupported numbers.")
        for r in misses:
            why = []
            if r["missing_facts"]:
                why.append("missing " + ", ".join(r["missing_facts"]))
            if r["unsupported_numbers"]:
                why.append("unsupported " + ", ".join(r["unsupported_numbers"]))
            lines += [f"- **{r['question']}** ({'; '.join(why)})", f"  - got: {_cell(r['prediction'])}",
                      f"  - approved: {r['reference']}"]
        lines.append("")
    text = "\n".join(lines)
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text

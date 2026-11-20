"""Streamlit app: ask the FAQ bot, browse the dataset, train / evaluate / merge, rate answers.

    streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # `streamlit run app/main.py` puts app/ on the path, not the project
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app import core, evaluate, llm, lora, storage  # noqa: E402

REPORTS = ROOT / "reports"
MODEL_LABELS = {"merged": "Merged LoRA model (deployment)", "adapter": "GPT-2 + LoRA adapter (unmerged)",
                "base": "GPT-2, no fine-tuning"}

st.set_page_config(page_title="Public FAQ Answer Bot", page_icon="🏛️", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_model(kind: str):
    model, tok, _ = llm.load_model(kind)
    return model, tok


@st.cache_data
def get_faqs():
    return core.load_faqs()


def read_report(name: str):
    path = REPORTS / name
    return storage.read_json(path) if path.exists() else None


faqs = get_faqs()
train_examples, held_out = core.split_examples(faqs)

# --- sidebar ------------------------------------------------------------------
with st.sidebar:
    st.header("Model")
    kinds = llm.available_models()
    kind = st.radio("Answer with", kinds, format_func=MODEL_LABELS.get)
    if kinds == ["base"]:
        st.warning("No adapter trained yet: answers come from plain GPT-2. Train one in the *Train* tab.")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1, help="0 = greedy, always the same answer")
    max_tokens = st.slider("Max answer tokens", 20, 120, 80, 10)
    st.divider()
    st.caption(f"{len(faqs)} FAQs from {len(core.department_counts(faqs))} departments of the "
               "fictional City of Riverton. Runs locally on GPT-2; no API key needed.")

st.title("🏛️ Public FAQ Answer Bot Trainer")
st.caption("Train GPT-2 with a hand-written LoRA adapter on departmental FAQ guidelines, check the "
           "answers, then merge the adapter for deployment.")

ask_tab, data_tab, train_tab, eval_tab, merge_tab, fb_tab = st.tabs(
    ["Ask the bot", "FAQ dataset", "Train", "Evaluation", "Merge for deployment", "Feedback"])

# --- ask --------------------------------------------------------------------------
with ask_tab:
    left, right = st.columns([3, 2])
    with right:
        st.subheader("Try a held-out question")
        st.caption("Paraphrases the model never saw in training.")
        dept = st.selectbox("Department", sorted(core.department_counts(faqs)))
        options = [e.question for e in held_out if e.department == dept]
        st.radio("Question", options, index=None, label_visibility="collapsed", key="picked",
                 on_change=lambda: st.session_state.update(question=st.session_state.picked or ""))
    with left:
        question = st.text_input("Ask a question about city services", key="question",
                                 placeholder="e.g. How much does it cost to register a car?")
        if st.button("Get answer", type="primary", disabled=not question.strip()):
            model, tok = get_model(kind)
            with st.spinner("Generating..."):
                answer = llm.generate_answer(model, tok, question, temperature=temperature,
                                             max_new_tokens=max_tokens, seed=None if temperature else 0)
            st.session_state["last"] = {"question": question, "answer": answer, "model": kind}

        last = st.session_state.get("last")
        if last:
            st.markdown(f"**Answer** ({MODEL_LABELS[last['model']]})")
            st.info(last["answer"] or "*(the model produced no answer)*")
            source = evaluate.closest_answer(last["answer"], faqs)
            hits = evaluate.fact_hits(last["answer"], source.facts)
            extra = evaluate.unsupported_numbers(last["answer"], source.answer)
            with st.expander(f"Check against the approved answer: {source.id} ({source.department})",
                             expanded=True):
                st.write(source.answer)
                st.write("Key facts: " + "  ".join(("✅ " if h else "❌ ") + f for f, h in zip(source.facts, hits)))
                if extra:
                    st.warning("Numbers not in the approved answer: " + ", ".join(sorted(extra)))

            with st.form("feedback", clear_on_submit=True):
                st.markdown("**Rate this answer** (administrators)")
                rating = st.select_slider("Quality", options=[1, 2, 3, 4, 5], value=4,
                                          format_func=lambda r: {1: "1 wrong", 2: "2 poor", 3: "3 partly right",
                                                                 4: "4 good", 5: "5 perfect"}[r])
                comment = st.text_input("Comment or corrected answer (optional)")
                if st.form_submit_button("Save rating"):
                    storage.record_feedback(last["question"], last["answer"], rating, comment, last["model"])
                    st.success("Saved to models/feedback.jsonl")

# --- dataset ----------------------------------------------------------------------
with data_tab:
    counts = core.department_counts(faqs)
    c1, c2, c3 = st.columns(3)
    c1.metric("FAQs", len(faqs))
    c2.metric("Training questions", len(train_examples))
    c3.metric("Held-out paraphrases", len(held_out))
    st.bar_chart(pd.Series(counts, name="FAQs"), horizontal=True)
    choice = st.multiselect("Filter departments", sorted(counts))
    rows = [{"id": f.id, "department": f.department, "question": f.question, "answer": f.answer,
             "paraphrases": " | ".join(f.paraphrases), "key facts": ", ".join(f.facts)}
            for f in faqs if not choice or f.department in choice]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("Edit `data/faqs.jsonl` to add FAQs. Every key fact must appear in its answer; the last "
               "paraphrase of each FAQ is held out for evaluation.")

# --- train ------------------------------------------------------------------------
with train_tab:
    st.subheader("LoRA fine-tuning")
    st.write("Only the low-rank matrices A and B are trained; GPT-2's 124M weights stay frozen.")
    c1, c2, c3, c4 = st.columns(4)
    r = c1.select_slider("Rank r", [1, 2, 4, 8, 16, 32, 64], value=16)
    alpha = c2.select_slider("Alpha", [1, 2, 4, 8, 16, 32, 64, 128], value=32)
    epochs = c3.slider("Epochs", 1, 30, 12)
    lr = c4.select_slider("Learning rate", [1e-4, 3e-4, 1e-3, 3e-3], value=1e-3, format_func=lambda x: f"{x:g}")
    targets = st.multiselect("Layers to adapt", lora.LoRAConfig().target_modules,
                             default=lora.LoRAConfig().target_modules)
    current = llm.adapter_dir() / "history.json"
    if current.exists():
        hist = storage.read_json(current)
        st.markdown(f"**Current adapter:** r={hist['config']['lora']['r']}, "
                    f"{hist['trainable_params']:,} trainable parameters "
                    f"({100 * hist['trainable_params'] / hist['total_params']:.2f}% of the model), "
                    f"best held-out loss {hist['best_val_loss']:.3f} at epoch {hist['best_epoch']}.")
        curve = pd.DataFrame(hist["epochs"]).set_index("epoch")[["train_loss", "val_loss"]]
        st.line_chart(curve.rename(columns={"train_loss": "train", "val_loss": "held-out"}))
    st.caption("Training takes a few minutes on a laptop CPU. It replaces `models/adapter`; merge again afterwards.")
    if st.button("Train adapter", type="primary", disabled=not targets):
        from app import train

        cfg = train.TrainConfig(epochs=epochs, lr=lr, lora=lora.LoRAConfig(r=r, alpha=alpha, target_modules=targets))
        with st.status("Training...", expanded=True) as status:
            model, tok = llm.load_base()
            hist = train.train_lora(model, tok, train_examples, held_out, cfg, device=str(model.device),
                                    out_dir=llm.adapter_dir(), log=st.write,
                                    extra_meta={"base_model": llm.base_model_name()})
            storage.write_json(llm.adapter_dir() / "history.json", hist)
            status.update(label="Training finished", state="complete")
        get_model.clear()
        st.rerun()

# --- evaluation ---------------------------------------------------------------------
with eval_tab:
    summary = read_report("eval_summary.json")
    if st.button("Run evaluation", help="Answers every FAQ with each system; takes a few minutes"):
        from app.cli import run_evaluation

        with st.status("Evaluating...", expanded=True):
            run_evaluation(log=st.write)
        st.rerun()
    if not summary:
        st.info("No evaluation yet. Run `python -m app.cli evaluate` or press the button.")
    else:
        st.write("**Fact recall**: share of key facts (fees, deadlines, phone numbers) present. "
                 "**Relevant**: the answer matches the right FAQ. **Unsupported**: states a number "
                 "not in the approved answer (lower is better).")
        for split, systems in summary["splits"].items():
            st.markdown(f"#### {split}")
            df = pd.DataFrame(systems).set_index("system")
            st.dataframe((df * 100).round(1).astype(str) + "%", use_container_width=True)
        st.markdown("#### Held-out answers from the LoRA model")
        rows = pd.DataFrame(summary["held_out_rows"])
        only_bad = st.checkbox("Only answers with missing facts or unsupported numbers")
        if only_bad:
            rows = rows[(rows.all_facts < 1) | (rows.unsupported > 0)]
        st.dataframe(rows[["department", "question", "prediction", "reference", "fact_recall", "missing_facts"]],
                     hide_index=True, use_container_width=True)
        sweep = REPORTS / "lora_sweep.md"
        if sweep.exists():
            with st.expander("LoRA configuration comparison (rank, alpha, layers)"):
                st.markdown(sweep.read_text(encoding="utf-8"))

# --- merge --------------------------------------------------------------------------
with merge_tab:
    st.subheader("Merge the adapter into the weights")
    st.latex(r"W' = W + \frac{\alpha}{r} B A")
    st.write("After merging, the model is a plain GPT-2 checkpoint: no LoRA code, no extra matmuls, "
             "loadable with `AutoModelForCausalLM.from_pretrained('models/merged')`.")
    report = read_report("merge_report.json")
    if report:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Layers merged", report["lora_layers_merged"])
        c2.metric("Max logit difference", f"{report['max_logit_diff']:.1e}")
        c3.metric("Identical answers", f"{report['identical_answers']}/{report['n_checked']}")
        c4.metric("Seconds per answer", f"{report['sec_per_answer_merged']:.2f}",
                  delta=f"{report['sec_per_answer_merged'] - report['sec_per_answer_adapter']:+.2f} vs adapter",
                  delta_color="inverse")
        st.write(f"Parameters: {report['params_with_adapter']:,} with adapter → {report['params_merged']:,} merged. "
                 f"Adapter on disk: {report['adapter_size_mb']} MB; merged model: {report['merged_size_mb']} MB.")
    disabled = "adapter" not in llm.available_models()
    if st.button("Merge adapter now", type="primary", disabled=disabled):
        from app import merge

        with st.status("Merging...", expanded=True):
            rep = merge.merge_and_save(log=st.write)
            storage.write_json(REPORTS / "merge_report.json", rep)
        get_model.clear()
        st.rerun()

# --- feedback -----------------------------------------------------------------------
with fb_tab:
    fb = storage.read_feedback()
    if not fb:
        st.info("No ratings yet. Rate answers in the *Ask the bot* tab.")
    else:
        df = pd.DataFrame(fb)
        c1, c2 = st.columns(2)
        c1.metric("Ratings", len(df))
        c2.metric("Average rating", f"{df.rating.mean():.2f} / 5")
        st.dataframe(df.groupby("model").rating.agg(["count", "mean"]).round(2), use_container_width=True)
        st.write("Low-rated answers (≤ 2): candidates to fix in `data/faqs.jsonl` before the next training run.")
        st.dataframe(df[df.rating <= 2][["time", "question", "answer", "rating", "comment"]],
                     hide_index=True, use_container_width=True)

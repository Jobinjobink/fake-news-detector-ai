"""SignalCheck AI — a transparent Streamlit fake-news pattern detector."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import streamlit as st

from train import METRICS_PATH, MODEL_PATH, train_and_save

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="SignalCheck AI", page_icon="◉", layout="wide")
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Space+Grotesk:wght@600;700&display=swap');
    .stApp { background: #0b0e13; color: #f5f0e8; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .hero { padding: 2.8rem 0 1.7rem; }
    .eyebrow { color:#ff6b57; font-weight:700; letter-spacing:.14em; font-size:.78rem; }
    .hero h1 { font-size:clamp(2.7rem,7vw,5.5rem); line-height:.92; margin:.5rem 0 1rem; }
    .hero p { color:#aeb5c1; max-width:720px; font-size:1.12rem; }
    .card { border:1px solid #262d38; background:#121721; border-radius:18px; padding:1.35rem; }
    .result-fake { border-left:5px solid #ff6b57; }
    .result-real { border-left:5px solid #43d6a1; }
    .score { font:700 3rem 'Space Grotesk'; }
    .muted { color:#9ca6b5; }
    div[data-testid="stMetric"] { background:#121721; border:1px solid #262d38; padding:1rem; border-radius:14px; }
    .stButton > button { background:#ff6b57; color:#0b0e13; border:0; font-weight:700; border-radius:10px; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Training the language model for its first launch…")
def load_model():
    if not (ROOT / MODEL_PATH.name).exists() or not (ROOT / METRICS_PATH.name).exists():
        return train_and_save(ROOT)
    model = joblib.load(ROOT / MODEL_PATH.name)
    metrics = json.loads((ROOT / METRICS_PATH.name).read_text(encoding="utf-8"))
    return model, metrics


model, metrics = load_model()

st.markdown(
    """<section class="hero"><div class="eyebrow">SIGNALCHECK / NLP LAB</div>
    <h1>Pause before<br>you share.</h1>
    <p>An interpretable machine-learning signal for English news headlines, trained on fact-check labels from PolitiFact and GossipCop.</p></section>""",
    unsafe_allow_html=True,
)

left, right = st.columns([1.55, 1], gap="large")
with left:
    st.subheader("Analyze a headline")
    headline = st.text_area(
        "News headline",
        placeholder="Paste a complete English news headline here…",
        height=145,
        label_visibility="collapsed",
    )
    analyze = st.button("Check the signal →", type="primary", use_container_width=True)
    if analyze:
        clean = headline.strip()
        if len(clean) < 12:
            st.warning("Please enter a fuller headline (at least 12 characters).")
        else:
            real_probability = float(model.predict_proba([clean])[0][1])
            fake_probability = 1 - real_probability
            is_fake = fake_probability >= 0.5
            label = "Potential misinformation pattern" if is_fake else "More consistent with credible-news patterns"
            score = fake_probability if is_fake else real_probability
            klass = "result-fake" if is_fake else "result-real"
            st.markdown(
                f"""<div class="card {klass}"><div class="muted">MODEL SIGNAL</div>
                <h3>{label}</h3><div class="score">{score:.0%}</div>
                <div class="muted">model confidence · not factual certainty</div></div>""",
                unsafe_allow_html=True,
            )
            st.progress(score)
            st.caption(f"Pattern scores — misinformation-like {fake_probability:.1%} · credible-news-like {real_probability:.1%}")

with right:
    st.markdown("<div class='card'><h3>How to use this</h3><p class='muted'>Treat the result as a reason to investigate—not a verdict. Check the publisher, author, date, primary evidence, and independent reporting before sharing.</p></div>", unsafe_allow_html=True)
    st.write("")
    a, b = st.columns(2)
    a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
    b.metric("Macro F1", f"{metrics['f1_macro']:.1%}")
    a.metric("Training rows", f"{metrics['train_records']:,}")
    b.metric("Test rows", f"{metrics['test_records']:,}")

st.divider()
with st.expander("Model card & limitations"):
    st.markdown(
        f"""
        **Model:** TF-IDF word and phrase features + class-balanced logistic regression.  
        **Dataset:** {metrics['dataset']} ({metrics['records']:,} deduplicated headlines).  
        **Evaluation:** fixed 80/20 stratified split; accuracy {metrics['accuracy']:.1%}, macro F1 {metrics['f1_macro']:.1%}.

        This is a portfolio demonstration, **not a fact-checking service**. It reads language patterns only; it does not browse sources, inspect evidence, or know current events. Dataset topics, publishers, dates, and labeling methods can create bias. Satire, breaking news, non-English text, and unfamiliar domains may be misclassified.
        """
    )

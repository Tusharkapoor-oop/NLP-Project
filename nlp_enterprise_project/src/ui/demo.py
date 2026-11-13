"""
NLP Text Classifier Demo
========================
A friendly, no-AI-vibe Streamlit dashboard to try both production models.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import requests
import streamlit as st

# ----------------------- page setup -----------------------
st.set_page_config(
    page_title="News Classifier Demo",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------- sidebar -----------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2964/2964514.png", width=80)
    st.title("News Classifier")
    st.markdown("""
    **About this demo**
    
    This little app lets you paste any news headline or article and see how two different NLP models classify it.
    
    - 🧠 **Classic pipeline**: Word2Vec + BiLSTM trained on BBC News
    - 🤖 **Transformer pipeline**: DistilBERT fine-tuned on AG News
    """)
    
    st.divider()
    st.subheader("Model status")
    w2v_status = "✅ online" if requests.get("http://localhost:8001/health", timeout=2).status_code == 200 else "⚠️ offline"
    bert_status = "✅ online" if requests.get("http://localhost:8002/health", timeout=2).status_code == 200 else "⚠️ offline"
    st.markdown(f"- BBC classifier API: **{w2v_status}** (port 8001)")
    st.markdown(f"- AG News classifier API: **{bert_status}** (port 8002)")
    st.divider()
    st.caption("Built for a final-year NLP project. Clean, simple, human-friendly.")

# ----------------------- header -----------------------
st.title("📰 News Classifier Demo")
st.markdown("""
Welcome! Paste a news snippet below and choose which model should read it.
Each model was trained on trusted Hugging Face datasets.
""")

# ----------------------- shared text input -----------------------
with st.container(border=True):
    text_input = st.text_area(
        "Paste your news text here",
        placeholder="Example: \"The football team won the final after a tense penalty shootout...\"",
        height=160,
        label_visibility="collapsed",
    )
    col1, col2, col3 = st.columns([1, 1, 3])
    with col1:
        k = st.slider("How many top predictions to show?", 1, 5, 3, key="topk")

# ----------------------- helper -----------------------
def call_api(url: str, text: str, top_k: int):
    try:
        response = requests.post(url, json={"texts": [text], "top_k": top_k}, timeout=30)
        response.raise_for_status()
        return response.json()["predictions"][0]["top_predictions"]
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the API. Make sure it is running on the port shown in the sidebar.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {e}")
        return None


def show_predictions(preds, color: str):
    if not preds:
        return
    labels = [p["label"] for p in preds]
    confidences = [p["confidence"] * 100 for p in preds]
    
    st.markdown("#### Top predictions")
    cols = st.columns(len(preds))
    for i, p in enumerate(preds):
        with cols[i]:
            st.container(border=True)
            st.markdown(f"**{p['label']}**")
            st.markdown(f"<span style='font-size:1.4rem; color:{color}; font-weight:bold'>{p['confidence']:.1%}</span>", unsafe_allow_html=True)
            st.progress(int(p["confidence"] * 100), text=f"{p['confidence']:.1%} confidence")
    
    # bar chart for a nicer visual
    st.markdown("#### Confidence breakdown")
    chart_data = {p["label"]: p["confidence"] for p in preds}
    st.bar_chart(chart_data, color=color, height=200)

# ----------------------- action buttons -----------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🧠 BBC News classifier")
    st.caption("Word2Vec + BiLSTM — categories: business, entertainment, politics, sport, tech")
    if st.button("Classify with Word2Vec + BiLSTM", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("Reading the article the old-school way..."):
                preds = call_api("http://localhost:8001/predict", text_input, k)
            if preds:
                show_predictions(preds, "#4CAF50")
        else:
            st.warning("Please paste some text first.")

with col2:
    st.markdown("### 🤖 AG News classifier")
    st.caption("DistilBERT — categories: World, Sports, Business, Sci/Tech")
    if st.button("Classify with DistilBERT", type="primary", use_container_width=True):
        if text_input.strip():
            with st.spinner("Transformers doing their thing..."):
                preds = call_api("http://localhost:8002/predict", text_input, k)
            if preds:
                show_predictions(preds, "#2196F3")
        else:
            st.warning("Please paste some text first.")

# ----------------------- footer -----------------------
st.divider()
st.markdown("""
<small>
Made for a final-year NLP project. Datasets: 
<a href="https://huggingface.co/datasets/SetFit/bbc-news" target="_blank">SetFit/bbc-news</a> and 
<a href="https://huggingface.co/datasets/SetFit/ag_news" target="_blank">SetFit/ag_news</a>.
</small>
""", unsafe_allow_html=True)

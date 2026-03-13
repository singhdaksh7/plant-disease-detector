"""
🌿 Plant Disease Detector — Premium Edition
AI-powered plant disease diagnosis from leaf images.
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from utils.disease_info import DISEASE_INFO, CLASS_NAMES, get_disease_info, get_severity_color
import time

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/plant_disease_model.keras")

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector | AI Diagnosis",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────
# Premium CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@400;600;700&display=swap');

    :root {
        --bg-primary: #0a1a0f;
        --bg-secondary: #0f261a;
        --bg-card: rgba(15, 38, 26, 0.6);
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.06);
        --text-primary: #e8f5e9;
        --text-secondary: #a5d6a7;
        --text-muted: #6b9e7a;
        --accent-green: #4caf50;
        --accent-emerald: #2e7d32;
        --accent-lime: #8bc34a;
        --accent-gold: #ffd54f;
        --accent-red: #ef5350;
        --accent-orange: #ff9800;
        --gradient-primary: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        --gradient-glow: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%);
        --shadow-glow: 0 0 40px rgba(76, 175, 80, 0.15);
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--bg-primary);
        background-image:
            radial-gradient(ellipse at 20% 50%, rgba(46, 125, 50, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(139, 195, 74, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(76, 175, 80, 0.04) 0%, transparent 50%);
        font-family: 'DM Sans', sans-serif;
    }

    .stApp * {
        color: var(--text-primary) !important;
        font-family: 'DM Sans', sans-serif !important;
    }

    /* Hide Streamlit defaults */
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {
        padding-top: 2rem !important;
        max-width: 1200px !important;
    }

    /* ── Hero Section ── */
    .hero {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        position: relative;
    }
    .hero::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(76, 175, 80, 0.1) 0%, transparent 70%);
        border-radius: 50%;
        z-index: 0;
    }
    .hero-icon {
        font-size: 4rem;
        margin-bottom: 0.5rem;
        display: block;
        position: relative;
        z-index: 1;
        animation: float 3s ease-in-out infinite;
    }
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }
    .hero h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 3.2rem !important;
        font-weight: 700 !important;
        background: linear-gradient(135deg, #e8f5e9 0%, #a5d6a7 50%, #81c784 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 !important;
        position: relative;
        z-index: 1;
        letter-spacing: -0.5px;
    }
    .hero-subtitle {
        font-size: 1.15rem !important;
        color: var(--text-muted) !important;
        margin-top: 0.8rem;
        position: relative;
        z-index: 1;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    /* ── Stats Bar ── */
    .stats-bar {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin: 2rem 0 2.5rem 0;
        position: relative;
        z-index: 1;
    }
    .stat-item {
        text-align: center;
    }
    .stat-number {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: var(--accent-green) !important;
        display: block;
    }
    .stat-label {
        font-size: 0.8rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-top: 2px;
    }

    /* ── Glass Card ── */
    .glass-card {
        background: var(--bg-glass);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(76, 175, 80, 0.15);
        box-shadow: var(--shadow-glow);
    }

    /* ── Section Headers ── */
    .section-header {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: var(--text-primary) !important;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* ── Result Banner ── */
    .result-banner {
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    .result-banner::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        opacity: 0.1;
        z-index: 0;
    }
    .result-healthy {
        background: linear-gradient(135deg, rgba(46, 125, 50, 0.2) 0%, rgba(76, 175, 80, 0.1) 100%);
        border: 1px solid rgba(76, 175, 80, 0.2);
    }
    .result-diseased {
        background: linear-gradient(135deg, rgba(239, 83, 80, 0.15) 0%, rgba(255, 152, 0, 0.1) 100%);
        border: 1px solid rgba(239, 83, 80, 0.2);
    }
    .result-title {
        font-family: 'Playfair Display', serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.5rem 0 !important;
        position: relative;
        z-index: 1;
    }
    .result-plant {
        font-size: 1rem !important;
        color: var(--text-secondary) !important;
        position: relative;
        z-index: 1;
        margin-bottom: 0.3rem;
    }
    .result-description {
        font-size: 0.95rem !important;
        color: var(--text-muted) !important;
        line-height: 1.6;
        position: relative;
        z-index: 1;
        margin-top: 1rem;
    }

    /* ── Severity Chip ── */
    .severity-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 16px;
        border-radius: 50px;
        font-size: 0.8rem !important;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        position: relative;
        z-index: 1;
    }
    .severity-none { background: rgba(76, 175, 80, 0.15); border: 1px solid rgba(76, 175, 80, 0.3); }
    .severity-moderate { background: rgba(255, 213, 79, 0.15); border: 1px solid rgba(255, 213, 79, 0.3); }
    .severity-high { background: rgba(255, 152, 0, 0.15); border: 1px solid rgba(255, 152, 0, 0.3); }
    .severity-critical { background: rgba(239, 83, 80, 0.15); border: 1px solid rgba(239, 83, 80, 0.3); }

    /* ── Confidence Bars ── */
    .confidence-row {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.8rem 0;
        border-bottom: 1px solid var(--border-glass);
    }
    .confidence-row:last-child { border-bottom: none; }
    .confidence-rank {
        font-family: 'Playfair Display', serif !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        width: 35px;
        text-align: center;
        opacity: 0.6;
    }
    .confidence-info { flex: 1; }
    .confidence-label {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: var(--text-primary) !important;
        margin-bottom: 4px;
    }
    .confidence-track {
        width: 100%;
        height: 8px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-out;
    }
    .confidence-value {
        font-size: 1rem !important;
        font-weight: 600 !important;
        min-width: 55px;
        text-align: right;
    }

    /* ── Info Cards ── */
    .info-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-glass);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }
    .info-card:hover {
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-2px);
    }
    .info-card-header {
        font-size: 0.75rem !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: var(--text-muted) !important;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    /* ── Symptom Pills ── */
    .symptom-pill {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 0.7rem 1rem;
        background: rgba(239, 83, 80, 0.05);
        border: 1px solid rgba(239, 83, 80, 0.1);
        border-radius: 12px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem !important;
        line-height: 1.5;
        transition: all 0.2s ease;
    }
    .symptom-pill:hover {
        background: rgba(239, 83, 80, 0.08);
        transform: translateX(4px);
    }
    .symptom-dot {
        width: 8px; height: 8px; min-width: 8px;
        background: var(--accent-red);
        border-radius: 50%;
        margin-top: 7px;
    }

    /* ── Treatment Cards ── */
    .treatment-card {
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 0.9rem 1.2rem;
        background: rgba(76, 175, 80, 0.05);
        border: 1px solid rgba(76, 175, 80, 0.1);
        border-radius: 12px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem !important;
        line-height: 1.5;
        transition: all 0.2s ease;
    }
    .treatment-card:hover {
        background: rgba(76, 175, 80, 0.08);
        transform: translateX(4px);
    }
    .treatment-num {
        background: rgba(76, 175, 80, 0.15);
        color: var(--accent-green) !important;
        width: 26px; height: 26px; min-width: 26px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem !important;
        font-weight: 700;
    }

    /* ── Tips Grid ── */
    .tips-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.8rem;
        margin-top: 1rem;
    }
    .tip-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border-glass);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    .tip-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(76, 175, 80, 0.15);
    }
    .tip-icon { font-size: 1.8rem; margin-bottom: 0.5rem; display: block; }
    .tip-title {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.2rem;
    }
    .tip-desc {
        font-size: 0.75rem !important;
        color: var(--text-muted) !important;
        line-height: 1.4;
    }

    /* ── Footer ── */
    .app-footer {
        text-align: center;
        padding: 3rem 0 2rem 0;
        border-top: 1px solid var(--border-glass);
        margin-top: 3rem;
    }
    .footer-text {
        font-size: 0.8rem !important;
        color: var(--text-muted) !important;
        letter-spacing: 0.5px;
    }
    .footer-brand {
        font-family: 'Playfair Display', serif !important;
        font-size: 1rem !important;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    /* ── Streamlit Overrides ── */
    .stFileUploader > div {
        background: rgba(76, 175, 80, 0.03) !important;
        border: 2px dashed rgba(76, 175, 80, 0.25) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        transition: all 0.3s ease !important;
    }
    .stFileUploader > div:hover {
        border-color: rgba(76, 175, 80, 0.4) !important;
        background: rgba(76, 175, 80, 0.06) !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent !important;
        gap: 0.5rem;
        border-bottom: 1px solid var(--border-glass) !important;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-muted) !important;
        border-radius: 10px 10px 0 0 !important;
        padding: 0.7rem 1.5rem !important;
        font-weight: 500 !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary) !important;
        background: rgba(255, 255, 255, 0.03) !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(76, 175, 80, 0.1) !important;
        color: var(--accent-green) !important;
        border-bottom: 2px solid var(--accent-green) !important;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none !important; }

    .stImage { border-radius: 16px; overflow: hidden; }

    /* ── Animations ── */
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-slide { animation: slideUp 0.6s ease-out; }
    .animate-fade { animation: fadeIn 0.8s ease-out; }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: rgba(76, 175, 80, 0.3); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(76, 175, 80, 0.5); }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model Loading (Cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


def predict(model, image: Image.Image) -> list:
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:3]
    return [(CLASS_NAMES[idx], float(predictions[idx])) for idx in top_indices]


def get_severity_class(severity):
    return {"None": "severity-none", "Moderate": "severity-moderate",
            "High": "severity-high", "Critical": "severity-critical"}.get(severity, "severity-none")


def get_confidence_color(index):
    colors = ["linear-gradient(90deg, #2e7d32, #4caf50)",
              "linear-gradient(90deg, #1565c0, #42a5f5)",
              "linear-gradient(90deg, #6a1b9a, #ab47bc)"]
    return colors[min(index, len(colors)-1)]


def display_results(results):
    top_class, top_confidence = results[0]
    disease_data = get_disease_info(top_class)
    if not disease_data:
        st.error("Disease information not found.")
        return

    severity = disease_data["severity"]
    is_healthy = severity == "None"
    banner_class = "result-healthy" if is_healthy else "result-diseased"
    icon = "✅" if is_healthy else "⚠️"
    title = f"{disease_data['plant']} — Healthy" if is_healthy else disease_data['disease']
    severity_class = get_severity_class(severity)
    severity_label = "No Disease" if is_healthy else severity

    st.markdown(f"""
    <div class="result-banner {banner_class} animate-slide">
        <div class="result-plant">{icon} Detected on <strong>{disease_data['plant']}</strong></div>
        <div class="result-title">{title}</div>
        <span class="severity-chip {severity_class}">● {severity_label}</span>
        <span style="margin-left: 12px; font-size: 0.9rem; opacity: 0.7;">
            {top_confidence*100:.1f}% confidence
        </span>
        <div class="result-description">{disease_data['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Top 3 Predictions
    st.markdown('<div class="section-header">📊 Prediction Confidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for i, (class_name, conf) in enumerate(results):
        info = get_disease_info(class_name)
        label = f"{info['plant']} → {info['disease']}" if info else class_name
        pct = conf * 100
        gradient = get_confidence_color(i)
        color = ['#4caf50', '#64b5f6', '#ab47bc'][min(i, 2)]
        st.markdown(f"""
        <div class="confidence-row">
            <div class="confidence-rank">{i+1}</div>
            <div class="confidence-info">
                <div class="confidence-label">{label}</div>
                <div class="confidence-track">
                    <div class="confidence-fill" style="width: {pct}%; background: {gradient};"></div>
                </div>
            </div>
            <div class="confidence-value" style="color: {color} !important;">{pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(["🔍 Symptoms", "💊 Treatment & Remedies"])
    with tab1:
        if is_healthy:
            st.markdown("""
            <div class="info-card animate-fade">
                <div style="text-align: center; padding: 1.5rem 0;">
                    <div style="font-size: 3rem; margin-bottom: 0.8rem;">🌱</div>
                    <div style="font-size: 1.2rem; font-weight: 600; color: #4caf50 !important; margin-bottom: 0.5rem;">
                        Your plant looks healthy!</div>
                    <div style="font-size: 0.9rem; color: #6b9e7a !important;">
                        No disease symptoms were detected. Keep up the great care!</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card animate-fade">', unsafe_allow_html=True)
            st.markdown('<div class="info-card-header">🔎 Identified Symptoms</div>', unsafe_allow_html=True)
            for s in disease_data["symptoms"]:
                st.markdown(f'<div class="symptom-pill"><div class="symptom-dot"></div><div>{s}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-card animate-fade">', unsafe_allow_html=True)
        header = "🌱 Preventive Care Tips" if is_healthy else "💚 Recommended Treatment Plan"
        st.markdown(f'<div class="info-card-header">{header}</div>', unsafe_allow_html=True)
        for i, t in enumerate(disease_data["treatment"], 1):
            st.markdown(f"""
            <div class="treatment-card">
                <div class="treatment-num">{i}</div>
                <div>{t}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────
# ███  MAIN APP LAYOUT
# ──────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <span class="hero-icon">🌿</span>
    <h1>Plant Disease Detector</h1>
    <div class="hero-subtitle">AI-powered diagnosis from a single leaf image</div>
</div>
""", unsafe_allow_html=True)

# Stats
st.markdown("""
<div class="stats-bar">
    <div class="stat-item"><span class="stat-number">38</span><span class="stat-label">Classes</span></div>
    <div class="stat-item"><span class="stat-number">14</span><span class="stat-label">Crops</span></div>
    <div class="stat-item"><span class="stat-number">26</span><span class="stat-label">Diseases</span></div>
    <div class="stat-item"><span class="stat-number">96%+</span><span class="stat-label">Accuracy</span></div>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 2rem;">
        <div style="font-size: 2.5rem; margin-bottom: 1rem;">🧪</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #ffd54f !important; margin-bottom: 0.5rem;">
            Demo Mode Active</div>
        <div style="font-size: 0.85rem; color: #6b9e7a !important; line-height: 1.6;">
            No trained model found. Upload an image to see a simulated diagnosis.<br>
            To use the real model, run: <code style="color: #81c784 !important;">python train.py --data_dir ./data/PlantVillage</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Upload & Preview
col_left, col_spacer, col_right = st.columns([5, 0.5, 5])

with col_left:
    st.markdown('<div class="section-header">📤 Upload Leaf Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drag & drop or click to browse",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of a single plant leaf for best results.",
        label_visibility="collapsed"
    )
    st.markdown("""
    <div class="tips-grid">
        <div class="tip-card"><span class="tip-icon">📸</span><div class="tip-title">Clear & Bright</div><div class="tip-desc">Use well-lit, focused images</div></div>
        <div class="tip-card"><span class="tip-icon">🍃</span><div class="tip-title">Single Leaf</div><div class="tip-desc">Focus on one leaf at a time</div></div>
        <div class="tip-card"><span class="tip-icon">🔍</span><div class="tip-title">Show Symptoms</div><div class="tip-desc">Capture visible spots or marks</div></div>
        <div class="tip-card"><span class="tip-icon">🚫</span><div class="tip-title">Avoid Blur</div><div class="tip-desc">Hold steady, no motion blur</div></div>
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="section-header">🖼️ Image Preview</div>', unsafe_allow_html=True)
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)
    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 4rem 2rem; min-height: 300px;
                    display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="font-size: 3.5rem; opacity: 0.3; margin-bottom: 1rem;">🍂</div>
            <div style="font-size: 1rem; color: #6b9e7a !important;">Your image will appear here</div>
        </div>
        """, unsafe_allow_html=True)

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if model is not None:
        with st.spinner("🔍 Analyzing leaf image..."):
            results = predict(model, image)
            display_results(results)
    else:
        with st.spinner("🔍 Simulating analysis..."):
            time.sleep(1.5)
            display_results([
                ("Tomato___Early_blight", 0.87),
                ("Tomato___Late_blight", 0.08),
                ("Tomato___Septoria_leaf_spot", 0.03)
            ])

# Supported Plants
st.markdown("---")
st.markdown('<div class="section-header" style="margin-top: 2rem;">🌱 Supported Plants</div>', unsafe_allow_html=True)

plants_data = {}
for key, val in DISEASE_INFO.items():
    plant = val["plant"]
    disease = val["disease"]
    if plant not in plants_data:
        plants_data[plant] = []
    if disease != "Healthy":
        plants_data[plant].append(disease)

plant_icons = {"Apple": "🍎", "Blueberry": "🫐", "Cherry": "🍒", "Corn (Maize)": "🌽",
               "Grape": "🍇", "Orange": "🍊", "Peach": "🍑", "Bell Pepper": "🫑",
               "Potato": "🥔", "Raspberry": "🫐", "Soybean": "🫘", "Squash": "🎃",
               "Strawberry": "🍓", "Tomato": "🍅"}

cols = st.columns(4)
for i, (plant, diseases) in enumerate(sorted(plants_data.items())):
    with cols[i % 4]:
        icon = plant_icons.get(plant, "🌱")
        dc = len(diseases)
        dt = f"{dc} disease{'s' if dc != 1 else ''}" if dc > 0 else "Health check only"
        st.markdown(f"""
        <div class="info-card" style="text-align: center; padding: 1.2rem;">
            <div style="font-size: 2rem; margin-bottom: 0.4rem;">{icon}</div>
            <div style="font-size: 0.9rem; font-weight: 600;">{plant}</div>
            <div style="font-size: 0.75rem; color: #6b9e7a !important;">{dt}</div>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="app-footer">
    <div class="footer-brand">🌿 Plant Disease Detector</div>
    <div class="footer-text">Built with TensorFlow & Streamlit · MobileNetV2 Transfer Learning · PlantVillage Dataset</div>
    <div class="footer-text" style="margin-top: 0.5rem; opacity: 0.5;">
        For educational purposes only — not a substitute for professional agricultural advice</div>
</div>
""", unsafe_allow_html=True)

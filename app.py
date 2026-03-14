"""
🌾 FASAL — AI-Driven Crop Disease Detector
AI-powered plant disease diagnosis from leaf images.
"""

import os
import io
import json
import base64
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from utils.disease_info import DISEASE_INFO, CLASS_NAMES, get_disease_info, get_severity_color
import time
from datetime import datetime
import requests
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/plant_disease_model.keras")
# Replace with your Google Apps Script URL after setup
GOOGLE_SHEET_URL = os.environ.get("GOOGLE_SHEET_URL", "https://script.google.com/macros/s/AKfycbza2d7JheJNuYDuOPA-iDGzEGpaatbQpYW3n-6Ii5MSr-T2cmYrcHejfULzhwrP28l3eA/exec")

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="FASAL | AI-Driven Crop Disease Detector",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ──────────────────────────────────────────────
# Premium CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=Playfair+Display:wght@400;600;700&display=swap');

    :root {
        --bg-primary: #0a1a0f;
        --bg-secondary: #0f261a;
        --bg-glass: rgba(255, 255, 255, 0.03);
        --border-glass: rgba(255, 255, 255, 0.06);
        --text-primary: #e8f5e9;
        --text-secondary: #a5d6a7;
        --text-muted: #6b9e7a;
        --accent-green: #4caf50;
        --accent-red: #ef5350;
        --shadow-glow: 0 0 40px rgba(76, 175, 80, 0.15);
    }

    .stApp {
        background: var(--bg-primary);
        background-image:
            radial-gradient(ellipse at 20% 50%, rgba(46, 125, 50, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 20%, rgba(139, 195, 74, 0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 80%, rgba(76, 175, 80, 0.04) 0%, transparent 50%);
        font-family: 'DM Sans', sans-serif;
    }
    .stApp * { color: var(--text-primary) !important; font-family: 'DM Sans', sans-serif !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 2rem !important; max-width: 1200px !important; }

    /* Hero */
    .hero { text-align: center; padding: 2.5rem 2rem 1rem; position: relative; }
    .hero::before {
        content: ''; position: absolute; top: 50%; left: 50%;
        transform: translate(-50%, -50%); width: 400px; height: 400px;
        background: radial-gradient(circle, rgba(76,175,80,0.1) 0%, transparent 70%);
        border-radius: 50%; z-index: 0;
    }
    .hero-icon { font-size: 3.5rem; display: block; position: relative; z-index:1; animation: float 3s ease-in-out infinite; }
    @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }
    .hero h1 {
        font-family: 'Playfair Display', serif !important;
        font-size: 2.8rem !important; font-weight: 700 !important;
        background: linear-gradient(135deg, #e8f5e9, #a5d6a7, #81c784);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0.3rem 0 0 !important; position: relative; z-index: 1;
    }
    .hero-sub { font-size:1.05rem !important; color:var(--text-muted) !important; margin-top:0.5rem; position:relative; z-index:1; font-weight:300; letter-spacing:0.5px; }

    /* Stats */
    .stats-bar { display:flex; justify-content:center; gap:2.5rem; margin:1.5rem 0 2rem; position:relative; z-index:1; }
    .stat-item { text-align:center; }
    .stat-num { font-family:'Playfair Display',serif !important; font-size:1.8rem !important; font-weight:700 !important; color:var(--accent-green) !important; display:block; }
    .stat-lbl { font-size:0.75rem !important; color:var(--text-muted) !important; text-transform:uppercase; letter-spacing:2px; }

    /* Cards */
    .glass-card {
        background:var(--bg-glass); backdrop-filter:blur(20px);
        border:1px solid var(--border-glass); border-radius:20px;
        padding:2rem; margin-bottom:1.5rem; transition:all 0.3s ease;
    }
    .glass-card:hover { border-color:rgba(76,175,80,0.15); box-shadow:var(--shadow-glow); }
    .section-header {
        font-family:'Playfair Display',serif !important; font-size:1.4rem !important;
        font-weight:600 !important; margin-bottom:1rem; display:flex; align-items:center; gap:0.6rem;
    }

    /* Result Banner */
    .result-banner { border-radius:20px; padding:2rem 2.5rem; margin:1.5rem 0; overflow:hidden; }
    .result-healthy { background:linear-gradient(135deg, rgba(46,125,50,0.2), rgba(76,175,80,0.1)); border:1px solid rgba(76,175,80,0.2); }
    .result-diseased { background:linear-gradient(135deg, rgba(239,83,80,0.15), rgba(255,152,0,0.1)); border:1px solid rgba(239,83,80,0.2); }
    .result-title { font-family:'Playfair Display',serif !important; font-size:1.8rem !important; font-weight:700 !important; margin:0 0 0.5rem !important; }
    .result-plant { font-size:1rem !important; color:var(--text-secondary) !important; }
    .result-desc { font-size:0.9rem !important; color:var(--text-muted) !important; line-height:1.6; margin-top:0.8rem; }

    /* Severity */
    .sev-chip { display:inline-flex; align-items:center; gap:6px; padding:5px 14px; border-radius:50px; font-size:0.75rem !important; font-weight:600; letter-spacing:1px; text-transform:uppercase; }
    .sev-none { background:rgba(76,175,80,0.15); border:1px solid rgba(76,175,80,0.3); }
    .sev-moderate { background:rgba(255,213,79,0.15); border:1px solid rgba(255,213,79,0.3); }
    .sev-high { background:rgba(255,152,0,0.15); border:1px solid rgba(255,152,0,0.3); }
    .sev-critical { background:rgba(239,83,80,0.15); border:1px solid rgba(239,83,80,0.3); }

    /* Confidence */
    .conf-row { display:flex; align-items:center; gap:1rem; padding:0.7rem 0; border-bottom:1px solid var(--border-glass); }
    .conf-row:last-child { border-bottom:none; }
    .conf-rank { font-family:'Playfair Display',serif !important; font-size:1.3rem !important; font-weight:700 !important; width:30px; text-align:center; opacity:0.6; }
    .conf-info { flex:1; }
    .conf-label { font-size:0.9rem !important; font-weight:500 !important; margin-bottom:3px; }
    .conf-track { width:100%; height:7px; background:rgba(255,255,255,0.05); border-radius:10px; overflow:hidden; }
    .conf-fill { height:100%; border-radius:10px; }
    .conf-val { font-size:0.95rem !important; font-weight:600 !important; min-width:50px; text-align:right; }

    /* Info */
    .info-card { background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); border-radius:16px; padding:1.5rem; margin-bottom:0.8rem; transition:all 0.3s ease; }
    .info-card:hover { background:rgba(255,255,255,0.04); transform:translateY(-2px); }
    .info-card-hdr { font-size:0.75rem !important; text-transform:uppercase; letter-spacing:2px; color:var(--text-muted) !important; margin-bottom:1rem; font-weight:600; }

    /* Symptoms & Treatment */
    .sym-pill { display:flex; align-items:flex-start; gap:10px; padding:0.6rem 1rem; background:rgba(239,83,80,0.05); border:1px solid rgba(239,83,80,0.1); border-radius:12px; margin-bottom:0.4rem; font-size:0.88rem !important; line-height:1.5; transition:all 0.2s; }
    .sym-pill:hover { background:rgba(239,83,80,0.08); transform:translateX(4px); }
    .sym-dot { width:8px; height:8px; min-width:8px; background:var(--accent-red); border-radius:50%; margin-top:6px; }
    .tx-card { display:flex; align-items:flex-start; gap:12px; padding:0.8rem 1.1rem; background:rgba(76,175,80,0.05); border:1px solid rgba(76,175,80,0.1); border-radius:12px; margin-bottom:0.4rem; font-size:0.88rem !important; line-height:1.5; transition:all 0.2s; }
    .tx-card:hover { background:rgba(76,175,80,0.08); transform:translateX(4px); }
    .tx-num { background:rgba(76,175,80,0.15); color:var(--accent-green) !important; width:24px; height:24px; min-width:24px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:0.7rem !important; font-weight:700; }

    /* Tips */
    .tips-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:0.7rem; margin-top:0.8rem; }
    .tip-card { background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); border-radius:12px; padding:1rem; text-align:center; transition:all 0.3s; }
    .tip-card:hover { background:rgba(255,255,255,0.04); border-color:rgba(76,175,80,0.15); }
    .tip-icon { font-size:1.5rem; margin-bottom:0.3rem; display:block; }
    .tip-title { font-size:0.8rem !important; font-weight:600 !important; color:var(--text-secondary) !important; }
    .tip-desc { font-size:0.7rem !important; color:var(--text-muted) !important; }

    /* History */
    .hist-item { display:flex; align-items:center; gap:1rem; padding:0.8rem 1rem; background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); border-radius:12px; margin-bottom:0.5rem; }
    .hist-time { font-size:0.7rem !important; color:var(--text-muted) !important; min-width:60px; }
    .hist-result { flex:1; }
    .hist-disease { font-size:0.85rem !important; font-weight:600 !important; }
    .hist-conf { font-size:0.8rem !important; color:var(--accent-green) !important; }

    /* How It Works */
    .step-card { display:flex; align-items:flex-start; gap:1rem; padding:1.2rem; background:rgba(255,255,255,0.02); border:1px solid var(--border-glass); border-radius:16px; margin-bottom:0.8rem; transition:all 0.3s; }
    .step-card:hover { background:rgba(255,255,255,0.04); }
    .step-num { background:linear-gradient(135deg,#2e7d32,#4caf50); width:40px; height:40px; min-width:40px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:1.1rem !important; font-weight:700 !important; }
    .step-title { font-size:0.95rem !important; font-weight:600 !important; margin-bottom:0.2rem; }
    .step-desc { font-size:0.8rem !important; color:var(--text-muted) !important; line-height:1.5; }

    /* Contact Form */
    .contact-section { margin-top:2rem; }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #0f261a !important;
        background-color: #0f261a !important;
        border: 1px solid rgba(76,175,80,0.3) !important;
        border-radius: 12px !important;
        color: #e8f5e9 !important;
        -webkit-text-fill-color: #e8f5e9 !important;
        padding: 0.8rem 1rem !important;
        font-size: 0.9rem !important;
        caret-color: #4caf50 !important;
    }
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #6b9e7a !important;
        -webkit-text-fill-color: #6b9e7a !important;
        opacity: 1 !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        background: #0f261a !important;
        background-color: #0f261a !important;
        border-color: rgba(76,175,80,0.5) !important;
        box-shadow: 0 0 0 1px rgba(76,175,80,0.3) !important;
        color: #e8f5e9 !important;
        -webkit-text-fill-color: #e8f5e9 !important;
    }
    .stTextInput > div, .stTextArea > div {
        background: transparent !important;
    }
    .stTextInput label, .stTextArea label {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    /* Form container fix */
    [data-testid="stForm"] {
        background: transparent !important;
        border: none !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #2e7d32, #4caf50) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 20px rgba(76,175,80,0.3) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background:transparent !important; gap:0.5rem; border-bottom:1px solid var(--border-glass) !important; }
    .stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--text-muted) !important; border-radius:10px 10px 0 0 !important; padding:0.7rem 1.5rem !important; font-weight:500 !important; border:none !important; }
    .stTabs [data-baseweb="tab"]:hover { color:var(--text-primary) !important; background:rgba(255,255,255,0.03) !important; }
    .stTabs [aria-selected="true"] { background:rgba(76,175,80,0.1) !important; color:var(--accent-green) !important; border-bottom:2px solid var(--accent-green) !important; }
    .stTabs [data-baseweb="tab-panel"] { padding-top:1.5rem !important; }
    .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none !important; }

    .stImage { border-radius:16px; overflow:hidden; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background:var(--bg-secondary) !important; border-right:1px solid var(--border-glass) !important; }
    section[data-testid="stSidebar"] .stRadio label { color:var(--text-secondary) !important; }

    /* Footer */
    .app-footer { text-align:center; padding:3rem 0 2rem; border-top:1px solid var(--border-glass); margin-top:3rem; }
    .footer-brand { font-family:'Playfair Display',serif !important; font-size:1rem !important; color:var(--text-secondary) !important; margin-bottom:0.5rem; }
    .footer-text { font-size:0.8rem !important; color:var(--text-muted) !important; letter-spacing:0.5px; }

    /* Animations */
    @keyframes slideUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }
    .animate-slide { animation:slideUp 0.6s ease-out; }

    /* Scrollbar */
    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background:var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background:rgba(76,175,80,0.3); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────
VALIDATOR_PATH = os.environ.get("VALIDATOR_PATH", "models/leaf_validator.keras")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_validator():
    if not os.path.exists(VALIDATOR_PATH):
        return None
    try:
        return tf.keras.models.load_model(VALIDATOR_PATH)
    except Exception:
        return None


def preprocess_image(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(image).astype(np.float32), axis=0)
    # EfficientNetB0 has its own preprocessing (expects 0-255, normalizes internally)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array


def validate_leaf(validator, image, model=None):
    """
    Check if the uploaded image is actually a leaf using 3 methods:
    1. Color analysis — leaves should have significant green
    2. CNN validator (if available)
    3. Disease model confidence — if max prediction < threshold, it's not a leaf
    Returns (is_leaf: bool, confidence: float, reason: str)
    """
    img_np = np.array(image.convert("RGB").resize((224, 224)))
    r, g, b = img_np[:,:,0].mean(), img_np[:,:,1].mean(), img_np[:,:,2].mean()

    # ── Method 1: Color Analysis (most reliable for leaf vs non-leaf) ──
    total = r + g + b + 1e-8
    green_ratio = g / total
    red_ratio = r / total

    # Check for skin tones (high red, medium green, low-medium blue)
    is_skin = (r > 140 and g > 90 and b > 60 and r > g and r > b and (r - g) > 15)

    # Check for very low green (non-plant images)
    is_low_green = green_ratio < 0.30

    # Check if image is mostly gray/white/black (not colorful like leaves)
    std_color = np.std([r, g, b])
    is_gray = std_color < 15

    color_score = 1.0
    reason = ""
    if is_skin:
        color_score = 0.15
        reason = "Image appears to contain skin tones, not a plant leaf."
    elif is_gray:
        color_score = 0.20
        reason = "Image appears to be grayscale or lacks plant-like colors."
    elif is_low_green:
        color_score = 0.25
        reason = "Image lacks sufficient green color typical of plant leaves."

    # ── Method 2: CNN Validator ──
    cnn_score = 0.5
    if validator is not None:
        img_val = np.expand_dims(img_np / 255.0, axis=0)
        cnn_score = float(validator.predict(img_val, verbose=0)[0][0])

    # ── Method 3: Disease Model Confidence Check ──
    model_score = 0.5
    if model is not None:
        predictions = model.predict(preprocess_image(image), verbose=0)[0]
        max_conf = float(np.max(predictions))
        # If model is very uncertain (< 40%), likely not a leaf
        if max_conf < 0.40:
            model_score = 0.2
            if not reason:
                reason = "AI model could not confidently identify any known plant disease pattern."
        else:
            model_score = min(max_conf, 1.0)

    # ── Combined Decision ──
    # Color is strongest signal, then model confidence, then CNN
    combined = (0.45 * color_score + 0.35 * model_score + 0.20 * cnn_score)

    is_leaf = combined > 0.45
    if not reason and not is_leaf:
        reason = "Image does not appear to be a plant leaf."

    return is_leaf, combined, reason


def predict(model, image):
    predictions = model.predict(preprocess_image(image), verbose=0)[0]
    top_indices = np.argsort(predictions)[::-1][:3]
    return [(CLASS_NAMES[idx], float(predictions[idx])) for idx in top_indices]


# ──────────────────────────────────────────────
# Enhanced Grad-CAM with Bounding Box
# ──────────────────────────────────────────────
def generate_gradcam(model, image):
    """
    Generate enhanced Grad-CAM heatmap with automatic bounding box
    around the high-attention disease region.
    """
    try:
        img_array = preprocess_image(image)
        base_model = model.layers[0]

        # Find last conv layer
        last_conv = None
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break

        if last_conv is None:
            return None, None

        grad_model = tf.keras.Model(
            inputs=base_model.input,
            outputs=[base_model.get_layer(last_conv).output, base_model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions_out = grad_model(img_array)
            top_class = tf.argmax(predictions_out[0])
            classifier_input = tf.keras.Input(shape=conv_outputs.shape[1:])
            x = classifier_input
            for layer in model.layers[1:]:
                x = layer(x)
            classifier = tf.keras.Model(classifier_input, x)
            class_output = classifier(conv_outputs)
            loss = class_output[:, top_class]

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            return None, None

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_sum(conv_outputs[0] * pooled_grads, axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        # Resize heatmap
        heatmap_resized = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap_resized).resize((224, 224), Image.BILINEAR)
        heatmap_array = np.array(heatmap_img)

        # Apply colormap
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap_array / 255.0)[:, :, :3]
        heatmap_colored = np.uint8(heatmap_colored * 255)

        # Overlay on original
        original = image.convert("RGB").resize((224, 224))
        original_array = np.array(original)
        overlay = np.uint8(original_array * 0.55 + heatmap_colored * 0.45)

        # ── Generate Bounding Box around high-attention area ──
        threshold = 0.5
        binary_mask = (heatmap_array / 255.0) > threshold
        bbox_image = original.copy()

        if binary_mask.any():
            rows = np.any(binary_mask, axis=1)
            cols = np.any(binary_mask, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]

            # Add padding
            pad = 8
            y_min = max(0, y_min - pad)
            y_max = min(223, y_max + pad)
            x_min = max(0, x_min - pad)
            x_max = min(223, x_max + pad)

            # Draw bounding box on original image
            bbox_array = np.array(bbox_image)
            # Draw red rectangle (3px thick)
            for t in range(3):
                bbox_array[y_min+t, x_min:x_max+1] = [255, 50, 50]   # top
                bbox_array[y_max-t, x_min:x_max+1] = [255, 50, 50]   # bottom
                bbox_array[y_min:y_max+1, x_min+t] = [255, 50, 50]   # left
                bbox_array[y_min:y_max+1, x_max-t] = [255, 50, 50]   # right

            bbox_image = Image.fromarray(bbox_array)

        return Image.fromarray(overlay), bbox_image
    except Exception:
        return None, None


# ──────────────────────────────────────────────
# PDF Report Generator
# ──────────────────────────────────────────────
def generate_pdf_report(results, disease_data, image):
    """Generate a simple text-based diagnosis report."""
    top_class, top_conf = results[0]
    report = f"""
════════════════════════════════════════════════
    🌾 FASAL — CROP DISEASE DIAGNOSIS REPORT
════════════════════════════════════════════════

Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

─── DIAGNOSIS ─────────────────────────────────

Plant:       {disease_data['plant']}
Disease:     {disease_data['disease']}
Severity:    {disease_data['severity']}
Confidence:  {top_conf*100:.1f}%

─── DESCRIPTION ───────────────────────────────

{disease_data['description']}

─── SYMPTOMS ──────────────────────────────────

"""
    for i, s in enumerate(disease_data['symptoms'], 1):
        report += f"  {i}. {s}\n"

    report += "\n─── TREATMENT ─────────────────────────────────\n\n"
    for i, t in enumerate(disease_data['treatment'], 1):
        report += f"  {i}. {t}\n"

    report += f"""
─── ALL PREDICTIONS ───────────────────────────

"""
    for i, (cn, cf) in enumerate(results, 1):
        info = get_disease_info(cn)
        label = f"{info['plant']} - {info['disease']}" if info else cn
        report += f"  {i}. {label}: {cf*100:.2f}%\n"

    report += """
════════════════════════════════════════════════
    Generated by FASAL — AI-Driven Crop Disease Detector
    For educational purposes only
════════════════════════════════════════════════
"""
    return report


# ──────────────────────────────────────────────
# Google Sheets Submission
# ──────────────────────────────────────────────
def submit_to_sheets(name, email, message):
    """Submit contact form data to Google Sheets via Apps Script."""
    if not GOOGLE_SHEET_URL:
        return False, "Google Sheets URL not configured"
    try:
        payload = {
            "name": name,
            "email": email,
            "message": message,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        # Google Apps Script returns a 302 redirect on success
        # We need to follow it with allow_redirects=True
        response = requests.post(
            GOOGLE_SHEET_URL,
            json=payload,
            timeout=15,
            allow_redirects=True,
            headers={"Content-Type": "application/json"}
        )
        # Apps Script returns 200 after redirect, or 302 which means success
        if response.status_code in [200, 302]:
            return True, "Success"
        return False, f"Error: {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Request timed out"
    except Exception as e:
        return False, str(e)


# ──────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────
def get_sev_class(s):
    return {"None":"sev-none","Moderate":"sev-moderate","High":"sev-high","Critical":"sev-critical"}.get(s,"sev-none")

def get_conf_color(i):
    c = ["linear-gradient(90deg,#2e7d32,#4caf50)","linear-gradient(90deg,#1565c0,#42a5f5)","linear-gradient(90deg,#6a1b9a,#ab47bc)"]
    return c[min(i,2)]

def display_results(results, image=None, model=None):
    top_class, top_conf = results[0]
    dd = get_disease_info(top_class)
    if not dd:
        st.error("Disease info not found.")
        return

    sev = dd["severity"]
    healthy = sev == "None"
    banner = "result-healthy" if healthy else "result-diseased"
    icon = "✅" if healthy else "⚠️"
    title = f"{dd['plant']} — Healthy" if healthy else dd['disease']
    sc = get_sev_class(sev)
    sl = "No Disease" if healthy else sev

    # Save to history
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M"),
        "disease": title,
        "plant": dd["plant"],
        "confidence": f"{top_conf*100:.1f}%",
        "severity": sev
    })

    st.markdown(f"""
    <div class="result-banner {banner} animate-slide">
        <div class="result-plant">{icon} Detected on <strong>{dd['plant']}</strong></div>
        <div class="result-title">{title}</div>
        <span class="sev-chip {sc}">● {sl}</span>
        <span style="margin-left:12px; font-size:0.9rem; opacity:0.7;">{top_conf*100:.1f}% confidence</span>
        <div class="result-desc">{dd['description']}</div>
    </div>
    """, unsafe_allow_html=True)

    # Confidence bars
    st.markdown('<div class="section-header">📊 Prediction Confidence</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for i, (cn, cf) in enumerate(results):
        info = get_disease_info(cn)
        label = f"{info['plant']} → {info['disease']}" if info else cn
        pct = cf * 100
        grad = get_conf_color(i)
        clr = ['#4caf50','#64b5f6','#ab47bc'][min(i,2)]
        st.markdown(f"""
        <div class="conf-row">
            <div class="conf-rank">{i+1}</div>
            <div class="conf-info">
                <div class="conf-label">{label}</div>
                <div class="conf-track"><div class="conf-fill" style="width:{pct}%;background:{grad};"></div></div>
            </div>
            <div class="conf-val" style="color:{clr} !important;">{pct:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Tabs: Symptoms, Treatment, Grad-CAM, Report
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Symptoms", "💊 Treatment", "🧠 AI Focus (Grad-CAM)", "📄 Download Report"])

    with tab1:
        if healthy:
            st.markdown("""
            <div class="info-card"><div style="text-align:center;padding:1.5rem 0;">
                <div style="font-size:3rem;margin-bottom:0.8rem;">🌱</div>
                <div style="font-size:1.2rem;font-weight:600;color:#4caf50 !important;margin-bottom:0.5rem;">Your plant looks healthy!</div>
                <div style="font-size:0.9rem;color:#6b9e7a !important;">No disease symptoms detected. Keep up the great care!</div>
            </div></div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-card">', unsafe_allow_html=True)
            st.markdown('<div class="info-card-hdr">🔎 Identified Symptoms</div>', unsafe_allow_html=True)
            for s in dd["symptoms"]:
                st.markdown(f'<div class="sym-pill"><div class="sym-dot"></div><div>{s}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        hdr = "🌱 Preventive Care Tips" if healthy else "💚 Recommended Treatment Plan"
        st.markdown(f'<div class="info-card-hdr">{hdr}</div>', unsafe_allow_html=True)
        for i, t in enumerate(dd["treatment"], 1):
            st.markdown(f'<div class="tx-card"><div class="tx-num">{i}</div><div>{t}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="info-card-hdr">🧠 AI Explainability — Grad-CAM + Disease Region Detection</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size:0.85rem;color:#6b9e7a !important;margin-bottom:1rem;">See where the AI focused and the auto-detected disease region with bounding box. Red/yellow = high attention, blue = low attention.</p>', unsafe_allow_html=True)

        if model is not None and image is not None:
            with st.spinner("Generating Grad-CAM heatmap & disease region..."):
                heatmap, bbox_image = generate_gradcam(model, image)
                if heatmap and bbox_image:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Original Image**")
                        st.image(image.resize((224, 224)), use_container_width=True)
                    with col2:
                        st.markdown("**AI Focus Heatmap**")
                        st.image(heatmap, use_container_width=True)
                    with col3:
                        st.markdown("**Disease Region (Auto-detected)**")
                        st.image(bbox_image, use_container_width=True)
                    st.markdown("""
                    <div style="margin-top:1rem;padding:1rem;background:rgba(76,175,80,0.05);border:1px solid rgba(76,175,80,0.1);border-radius:12px;">
                        <div style="font-size:0.8rem;color:#a5d6a7 !important;line-height:1.6;">
                            <strong>How to read this:</strong><br>
                            🔴 <strong>Heatmap:</strong> Red/yellow regions = where the model looked most to identify the disease<br>
                            🟥 <strong>Bounding Box:</strong> Red rectangle = auto-detected area most likely showing disease symptoms
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("Grad-CAM visualization could not be generated for this image.")
        else:
            st.info("🧪 Grad-CAM requires the trained model to be loaded.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown('<div class="info-card-hdr">📄 Diagnosis Report</div>', unsafe_allow_html=True)
        report = generate_pdf_report(results, dd, image)
        st.download_button(
            label="⬇️ Download Report (.txt)",
            data=report,
            file_name=f"diagnosis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        with st.expander("Preview Report"):
            st.code(report, language=None)
        st.markdown('</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
# ███  SIDEBAR NAVIGATION
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:2rem;">🌾</div>
        <div style="font-family:'Playfair Display',serif !important; font-size:1.2rem; font-weight:600; margin-top:0.3rem;">
            FASAL
        </div>
        <div style="font-size:0.65rem;color:#6b9e7a !important;margin-top:0.2rem;letter-spacing:1px;">AI-DRIVEN CROP DISEASE DETECTOR</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home — Diagnose", "📊 History", "⚙️ How It Works", "🌱 Supported Plants", "📬 Contact Us", "ℹ️ About"],
        label_visibility="collapsed"
    )


# ════════════════════════════════════════════════
# ███  PAGE: HOME — DIAGNOSE
# ════════════════════════════════════════════════
if page == "🏠 Home — Diagnose":

    # Hero
    st.markdown("""
    <div class="hero">
        <span class="hero-icon">🌾</span>
        <h1>FASAL</h1>
        <div class="hero-sub">AI-Driven Crop Disease Detector — Diagnose from a single leaf image</div>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="stats-bar">
        <div class="stat-item"><span class="stat-num">38</span><span class="stat-lbl">Classes</span></div>
        <div class="stat-item"><span class="stat-num">14</span><span class="stat-lbl">Crops</span></div>
        <div class="stat-item"><span class="stat-num">26</span><span class="stat-lbl">Diseases</span></div>
        <div class="stat-item"><span class="stat-num">96%+</span><span class="stat-lbl">Accuracy</span></div>
    </div>
    """, unsafe_allow_html=True)

    model = load_model()

    if model is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:2rem;">
            <div style="font-size:2.5rem;margin-bottom:1rem;">🧪</div>
            <div style="font-size:1.1rem;font-weight:600;color:#ffd54f !important;margin-bottom:0.5rem;">Demo Mode Active</div>
            <div style="font-size:0.85rem;color:#6b9e7a !important;line-height:1.6;">
                No trained model found. Upload an image to see a simulated diagnosis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Upload + Preview
    col_l, col_s, col_r = st.columns([5, 0.5, 5])

    with col_l:
        st.markdown('<div class="section-header">📤 Upload Leaf Image</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")

        # Sample images section
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">🖼️ Or Try Sample Images</div>', unsafe_allow_html=True)
        sample_choice = st.selectbox("Select a sample plant disease:", [
            "None — I'll upload my own",
            "Tomato — Early Blight (Demo)",
            "Potato — Late Blight (Demo)",
            "Apple — Apple Scab (Demo)",
            "Grape — Black Rot (Demo)"
        ], label_visibility="collapsed")

        st.markdown("""
        <div class="tips-grid">
            <div class="tip-card"><span class="tip-icon">📸</span><div class="tip-title">Clear & Bright</div><div class="tip-desc">Well-lit, focused images</div></div>
            <div class="tip-card"><span class="tip-icon">🍃</span><div class="tip-title">Single Leaf</div><div class="tip-desc">One leaf at a time</div></div>
            <div class="tip-card"><span class="tip-icon">🔍</span><div class="tip-title">Show Symptoms</div><div class="tip-desc">Capture spots or marks</div></div>
            <div class="tip-card"><span class="tip-icon">🚫</span><div class="tip-title">Avoid Blur</div><div class="tip-desc">Hold steady, no blur</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">🖼️ Image Preview</div>', unsafe_allow_html=True)
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center;padding:4rem 2rem;min-height:300px;display:flex;flex-direction:column;justify-content:center;align-items:center;">
                <div style="font-size:3.5rem;opacity:0.3;margin-bottom:1rem;">🍂</div>
                <div style="font-size:1rem;color:#6b9e7a !important;">Your image will appear here</div>
            </div>
            """, unsafe_allow_html=True)

    # Run Prediction
    use_sample = sample_choice and "None" not in sample_choice
    if uploaded_file is not None or use_sample:
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

        if model is not None and image is not None:
            # ── Step 1: Leaf Validation Gate ──
            validator = load_validator()
            with st.spinner("🛡️ Validating image..."):
                is_leaf, leaf_score, reject_reason = validate_leaf(validator, image, model)

            if not is_leaf:
                # REJECTED — not a leaf
                st.markdown(f"""
                <div class="result-banner result-diseased animate-slide">
                    <div class="result-title">🚫 Not a Plant Leaf</div>
                    <div class="result-desc" style="margin-top:0.5rem;">
                        {reject_reason}
                        <br><br>
                        <strong>Leaf confidence score:</strong> {leaf_score*100:.1f}% (threshold: 45%)
                        <br><br>
                        <strong>Please upload:</strong><br>
                        ✅ A clear photo of a single plant leaf<br>
                        ❌ Not a human face, animal, object, or non-plant image
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # ACCEPTED — proceed with disease prediction
                with st.spinner("🔍 Analyzing leaf image..."):
                    results = predict(model, image)
                    display_results(results, image, model)
        else:
            with st.spinner("🔍 Simulating analysis..."):
                time.sleep(1)
                demo_map = {
                    "Tomato": [("Tomato___Early_blight", 0.87), ("Tomato___Late_blight", 0.08), ("Tomato___Septoria_leaf_spot", 0.03)],
                    "Potato": [("Potato___Late_blight", 0.91), ("Potato___Early_blight", 0.06), ("Tomato___Late_blight", 0.02)],
                    "Apple": [("Apple___Apple_scab", 0.89), ("Apple___Black_rot", 0.07), ("Apple___Cedar_apple_rust", 0.03)],
                    "Grape": [("Grape___Black_rot", 0.92), ("Grape___Esca_(Black_Measles)", 0.05), ("Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", 0.02)],
                }
                # Pick demo results
                demo_results = demo_map.get("Tomato")
                if use_sample:
                    for key in demo_map:
                        if key in sample_choice:
                            demo_results = demo_map[key]
                            break
                display_results(demo_results, image, None)


# ════════════════════════════════════════════════
# ███  PAGE: HISTORY
# ════════════════════════════════════════════════
elif page == "📊 History":
    st.markdown('<div class="section-header" style="font-size:1.8rem !important;">📊 Prediction History</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b9e7a !important;margin-bottom:1.5rem;">Track all predictions made during this session.</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:3rem;">
            <div style="font-size:3rem;margin-bottom:1rem;">📋</div>
            <div style="font-size:1.1rem;font-weight:500;margin-bottom:0.5rem;">No predictions yet</div>
            <div style="font-size:0.85rem;color:#6b9e7a !important;">Go to the Home page and upload a leaf image to start diagnosing.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Summary stats
        total = len(st.session_state.history)
        diseases = sum(1 for h in st.session_state.history if "Healthy" not in h["disease"])
        healthy_count = total - diseases

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2rem;">🔬</div>
                <div style="font-size:1.5rem;font-weight:700;color:#4caf50 !important;">{total}</div>
                <div style="font-size:0.75rem;color:#6b9e7a !important;text-transform:uppercase;letter-spacing:1px;">Total Scans</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2rem;">⚠️</div>
                <div style="font-size:1.5rem;font-weight:700;color:#ef5350 !important;">{diseases}</div>
                <div style="font-size:0.75rem;color:#6b9e7a !important;text-transform:uppercase;letter-spacing:1px;">Diseases Found</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:2rem;">✅</div>
                <div style="font-size:1.5rem;font-weight:700;color:#4caf50 !important;">{healthy_count}</div>
                <div style="font-size:0.75rem;color:#6b9e7a !important;text-transform:uppercase;letter-spacing:1px;">Healthy Plants</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")

        for h in st.session_state.history:
            sev_color = {"None":"#4caf50","Moderate":"#ffd54f","High":"#ff9800","Critical":"#ef5350"}.get(h["severity"],"#4caf50")
            st.markdown(f"""
            <div class="hist-item">
                <div class="hist-time">{h['time']}</div>
                <div class="hist-result">
                    <div class="hist-disease">{h['disease']}</div>
                    <div style="font-size:0.75rem;color:#6b9e7a !important;">{h['plant']}</div>
                </div>
                <div class="hist-conf">{h['confidence']}</div>
                <div style="width:10px;height:10px;border-radius:50%;background:{sev_color};"></div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()


# ════════════════════════════════════════════════
# ███  PAGE: HOW IT WORKS
# ════════════════════════════════════════════════
elif page == "⚙️ How It Works":
    st.markdown('<div class="section-header" style="font-size:1.8rem !important;">⚙️ How It Works</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b9e7a !important;margin-bottom:1.5rem;">Understanding the AI behind the plant disease detection.</p>', unsafe_allow_html=True)

    # Steps
    steps = [
        ("1", "📤", "Upload Image", "You upload a photo of a plant leaf through our web interface. The image can be in JPG, PNG, or WebP format."),
        ("2", "🛡️", "Leaf Validation Gate", "Before any disease analysis, a separate CNN classifier + color analysis checks if the image is actually a plant leaf. Non-leaf images (faces, objects, random photos) are rejected instantly. This prevents false diagnoses."),
        ("3", "🔄", "Image Preprocessing", "The validated leaf image is resized to 224×224 pixels and normalized. Pixel values are scaled to 0-1 range for optimal model performance."),
        ("4", "🧠", "EfficientNetB0 Prediction", "The image passes through EfficientNetB0, a state-of-the-art CNN using compound scaling, pre-trained on 14M ImageNet images and fine-tuned on 54,000+ PlantVillage leaf images."),
        ("5", "📊", "Classification", "The model outputs probability scores for all 38 classes. We extract the top 3 predictions with their confidence scores."),
        ("6", "🔍", "Grad-CAM + Disease Region Detection", "Gradient-weighted Class Activation Mapping generates a heatmap showing AI focus areas. An automatic bounding box is drawn around the high-attention disease region."),
        ("7", "📋", "Results & Treatment", "You receive the full diagnosis with disease details, symptoms, severity level, treatment recommendations, and a downloadable report.")
    ]

    for num, icon, title, desc in steps:
        st.markdown(f"""
        <div class="step-card">
            <div class="step-num">{num}</div>
            <div>
                <div class="step-title">{icon} {title}</div>
                <div class="step-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Architecture
    st.markdown("---")
    st.markdown('<div class="section-header">🏗️ Model Architecture</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
        <div class="info-card-hdr">Transfer Learning with EfficientNetB0 + Leaf Validation Gate</div>
        <div style="font-size:0.9rem; line-height:1.8; color:#a5d6a7 !important;">
            <strong>🛡️ Leaf Validator:</strong> Custom lightweight CNN (4 conv blocks) — binary classification (leaf vs not-leaf)<br>
            <strong>🧠 Disease Classifier:</strong> EfficientNetB0 (compound scaling — better than MobileNetV2)<br>
            <strong>Why EfficientNetB0?</strong> 77.1% ImageNet top-1 acc vs MobileNetV2's 71.8%, with only 5.3M params<br>
            <strong>Strategy:</strong> Two-phase training (Feature Extraction → Fine-Tuning)<br>
            <strong>Phase 1:</strong> Freeze base, train classifier head (lr=0.001)<br>
            <strong>Phase 2:</strong> Unfreeze top layers, fine-tune (lr=0.00001)<br>
            <strong>Data Augmentation:</strong> Rotation, flip, zoom, brightness, shear<br>
            <strong>Input Size:</strong> 224 × 224 × 3 (RGB)<br>
            <strong>Output:</strong> 38-class softmax probability distribution<br>
            <strong>Explainability:</strong> Grad-CAM heatmaps + auto bounding box<br>
            <strong>Validation Accuracy:</strong> 96%+
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Training Charts
    if os.path.exists("models/training_history.png"):
        st.markdown('<div class="section-header">📈 Training Performance</div>', unsafe_allow_html=True)
        st.image("models/training_history.png", use_container_width=True)


# ════════════════════════════════════════════════
# ███  PAGE: SUPPORTED PLANTS
# ════════════════════════════════════════════════
elif page == "🌱 Supported Plants":
    st.markdown('<div class="section-header" style="font-size:1.8rem !important;">🌱 Supported Plants & Diseases</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b9e7a !important;margin-bottom:1.5rem;">Our model can detect 26 diseases across 14 crop species.</p>', unsafe_allow_html=True)

    plant_icons = {"Apple":"🍎","Blueberry":"🫐","Cherry":"🍒","Corn (Maize)":"🌽",
                   "Grape":"🍇","Orange":"🍊","Peach":"🍑","Bell Pepper":"🫑",
                   "Potato":"🥔","Raspberry":"🫐","Soybean":"🫘","Squash":"🎃",
                   "Strawberry":"🍓","Tomato":"🍅"}

    plants_data = {}
    for key, val in DISEASE_INFO.items():
        p, d = val["plant"], val["disease"]
        if p not in plants_data:
            plants_data[p] = []
        if d != "Healthy":
            plants_data[p].append(d)

    for plant, diseases in sorted(plants_data.items()):
        icon = plant_icons.get(plant, "🌱")
        dc = len(diseases)

        with st.expander(f"{icon} {plant} — {dc} disease{'s' if dc != 1 else ''} detected" if dc > 0 else f"{icon} {plant} — Health check only"):
            if dc > 0:
                for d in diseases:
                    # Find full info
                    full_key = [k for k in DISEASE_INFO if DISEASE_INFO[k]["plant"] == plant and DISEASE_INFO[k]["disease"] == d]
                    if full_key:
                        info = DISEASE_INFO[full_key[0]]
                        sev = info["severity"]
                        sev_color = {"Moderate":"#ffd54f","High":"#ff9800","Critical":"#ef5350"}.get(sev,"#4caf50")
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:0.8rem;padding:0.5rem 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                            <div style="width:10px;height:10px;border-radius:50%;background:{sev_color};"></div>
                            <div style="flex:1;font-size:0.9rem;">{d}</div>
                            <div style="font-size:0.75rem;color:{sev_color} !important;font-weight:600;">{sev}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.markdown('<div style="font-size:0.9rem;color:#6b9e7a !important;">✅ Only healthy leaf detection available for this plant.</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
# ███  PAGE: CONTACT US
# ════════════════════════════════════════════════
elif page == "📬 Contact Us":
    st.markdown('<div class="section-header" style="font-size:1.8rem !important;">📬 Contact Us</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b9e7a !important;margin-bottom:1.5rem;">Have a question or found a bug? Send us a message and we\'ll get back to you!</p>', unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    with st.form("contact_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("👤 Your Name", placeholder="Enter your full name")
        with col2:
            email = st.text_input("📧 Email Address", placeholder="your@email.com")

        message = st.text_area("💬 Your Question / Doubt", placeholder="Describe your question, doubt, or feedback here...", height=150)

        submitted = st.form_submit_button("🚀 Send Message", use_container_width=True)

        if submitted:
            if not name or not email or not message:
                st.error("⚠️ Please fill in all fields!")
            elif "@" not in email or "." not in email:
                st.error("⚠️ Please enter a valid email address!")
            else:
                # Try Google Sheets submission
                if GOOGLE_SHEET_URL:
                    success, msg = submit_to_sheets(name, email, message)
                    if success:
                        st.success("✅ Message sent successfully! We'll get back to you soon.")
                    else:
                        st.warning(f"⚠️ Could not save to Google Sheets: {msg}. But your message has been logged locally.")
                        # Fallback: save locally
                        with open("contact_submissions.csv", "a") as f:
                            f.write(f'"{datetime.now()}","{name}","{email}","{message}"\n')
                        st.success("✅ Message saved locally!")
                else:
                    # Save locally as CSV
                    file_exists = os.path.exists("contact_submissions.csv")
                    with open("contact_submissions.csv", "a") as f:
                        if not file_exists:
                            f.write("Timestamp,Name,Email,Message\n")
                        f.write(f'"{datetime.now()}","{name}","{email}","{message}"\n')
                    st.success("✅ Message received! Thank you for reaching out.")

    st.markdown('</div>', unsafe_allow_html=True)

    # FAQ Section
    st.markdown("---")
    st.markdown('<div class="section-header">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)

    faqs = [
        ("What types of images work best?", "Clear, well-lit photos of a single plant leaf work best. Avoid blurry images or photos with multiple overlapping leaves."),
        ("How accurate is the model?", "Our EfficientNetB0 model achieves 96%+ validation accuracy on the PlantVillage dataset. It also includes a leaf validation gate that rejects non-leaf images. However, accuracy may vary with real-world images."),
        ("Can I use this for commercial farming?", "This tool is designed for educational purposes. For critical agricultural decisions, please consult a professional plant pathologist."),
        ("What plants are supported?", "We currently support 14 crop species including tomato, potato, apple, grape, corn, and more. Check the Supported Plants page for the full list."),
        ("Is my data stored?", "Uploaded images are processed in real-time and are not stored on our servers. Contact form submissions are saved for response purposes only."),
    ]

    for q, a in faqs:
        with st.expander(q):
            st.markdown(f'<div style="font-size:0.9rem;color:#a5d6a7 !important;line-height:1.6;">{a}</div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
# ███  PAGE: ABOUT
# ════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.markdown('<div class="section-header" style="font-size:1.8rem !important;">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:3rem;margin-bottom:0.5rem;">🌾</div>
            <div style="font-family:'Playfair Display',serif !important;font-size:1.5rem;font-weight:700;margin-bottom:0.5rem;">
                FASAL — AI-Driven Crop Disease Detector
            </div>
            <div style="font-size:0.9rem;color:#6b9e7a !important;line-height:1.6;max-width:600px;margin:0 auto;">
                FASAL (फसल = Crop) is an AI-powered web application that helps farmers, gardeners, and plant enthusiasts
                identify diseases in crop plants from leaf images. Built with modern deep learning
                techniques and designed for ease of use.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Developer Card
    st.markdown('<div class="section-header">👨‍💻 Developer</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="display:flex;align-items:center;gap:1.5rem;">
        <div style="width:80px;height:80px;min-width:80px;background:linear-gradient(135deg,#2e7d32,#4caf50);border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:2rem;">
            👨‍💻
        </div>
        <div>
            <div style="font-size:1.2rem;font-weight:700;">Daksh Singh</div>
            <div style="font-size:0.85rem;color:#6b9e7a !important;margin-top:0.3rem;">Machine Learning Developer</div>
            <div style="display:flex;gap:1rem;margin-top:0.8rem;">
                <a href="https://github.com/singhdaksh7" target="_blank" style="font-size:0.8rem;color:#4caf50 !important;text-decoration:none;font-weight:600;">
                    🔗 GitHub
                </a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech Stack
    st.markdown('<div class="section-header">🛠️ Tech Stack</div>', unsafe_allow_html=True)

    tech = [
        ("🧠", "TensorFlow / Keras", "Deep learning framework"),
        ("📱", "EfficientNetB0", "Pre-trained base model"),
        ("🎨", "Streamlit", "Web application framework"),
        ("🐳", "Docker", "Containerization"),
        ("☁️", "Render", "Cloud deployment"),
        ("📊", "scikit-learn", "Model evaluation"),
        ("🖼️", "Pillow", "Image processing"),
        ("📈", "Matplotlib", "Visualization"),
    ]

    cols = st.columns(4)
    for i, (icon, name, desc) in enumerate(tech):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="info-card" style="text-align:center;padding:1.2rem;">
                <div style="font-size:1.8rem;margin-bottom:0.4rem;">{icon}</div>
                <div style="font-size:0.85rem;font-weight:600;">{name}</div>
                <div style="font-size:0.7rem;color:#6b9e7a !important;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # Acknowledgments
    st.markdown("---")
    st.markdown('<div class="section-header">🙏 Acknowledgments</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="font-size:0.9rem;line-height:1.8;color:#a5d6a7 !important;">
        <strong>Dataset:</strong> PlantVillage by Abdallah Ali (54,000+ labeled images)<br>
        <strong>Base Model:</strong> EfficientNetB0 — Google Research<br>
        <strong>Framework:</strong> TensorFlow & Streamlit open-source community
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════
# ███  FOOTER (all pages)
# ════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
    <div class="footer-brand">🌾 FASAL — AI-Driven Crop Disease Detector</div>
    <div class="footer-text">Built with TensorFlow & Streamlit · EfficientNetB0 Transfer Learning · PlantVillage Dataset</div>
    <div class="footer-text" style="margin-top:0.5rem;opacity:0.5;">For educational purposes only — not a substitute for professional agricultural advice</div>
</div>
""", unsafe_allow_html=True)

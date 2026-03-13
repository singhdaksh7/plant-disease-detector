"""
🌿 Plant Disease Detector - Streamlit Web Application
Upload a plant leaf image and get instant disease diagnosis with
confidence scores, symptoms, and treatment recommendations.
"""

import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from utils.disease_info import DISEASE_INFO, CLASS_NAMES, get_disease_info, get_severity_color

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
MODEL_PATH = os.environ.get("MODEL_PATH", "models/plant_disease_model.keras")

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #2d6a4f 0%, #40916c 50%, #52b788 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        margin: 0;
    }
    .main-header p {
        color: #d8f3dc;
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }
    
    /* Result cards */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 1rem;
        border-left: 5px solid;
    }
    
    /* Severity badges */
    .severity-badge {
        display: inline-block;
        padding: 4px 16px;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
    }
    
    /* Confidence bar */
    .confidence-bar-container {
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 25px;
        margin: 5px 0;
    }
    .confidence-bar {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
        transition: width 0.5s ease;
    }
    
    /* Info sections */
    .info-section {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        margin: 0.5rem 0;
    }
    
    /* Symptom list */
    .symptom-item {
        padding: 0.3rem 0;
        border-bottom: 1px solid #e9ecef;
    }
    .symptom-item:last-child {
        border-bottom: none;
    }
    
    /* Treatment list */
    .treatment-item {
        background: #d8f3dc;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 0.3rem 0;
    }
    
    /* Top-3 prediction row */
    .pred-row {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        border-bottom: 1px solid #f0f0f0;
    }
    .pred-row:last-child {
        border-bottom: none;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 3px dashed #52b788;
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: #f0fdf4;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f0fdf4;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #6c757d;
        padding: 2rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Model Loading (Cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess uploaded image for model prediction."""
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(model, image: Image.Image) -> list:
    """
    Run prediction and return top-k results.
    
    Returns:
        List of tuples: [(class_name, confidence), ...]
    """
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)[0]
    
    # Get top 3 predictions
    top_indices = np.argsort(predictions)[::-1][:3]
    results = []
    for idx in top_indices:
        class_name = CLASS_NAMES[idx]
        confidence = float(predictions[idx])
        results.append((class_name, confidence))
    
    return results


def render_confidence_bar(confidence: float, color: str = "#52b788") -> str:
    """Generate HTML for a confidence bar."""
    percentage = confidence * 100
    return f"""
    <div class="confidence-bar-container">
        <div class="confidence-bar" style="width: {percentage}%; background: {color};">
            {percentage:.1f}%
        </div>
    </div>
    """


def display_results(results: list):
    """Display prediction results with all features."""
    top_class, top_confidence = results[0]
    disease_data = get_disease_info(top_class)
    
    if not disease_data:
        st.error("Disease information not found in database.")
        return
    
    severity = disease_data["severity"]
    severity_color = get_severity_color(severity)
    is_healthy = severity == "None"
    
    # ── Primary Result ──
    st.markdown("---")
    
    if is_healthy:
        st.markdown(f"""
        <div class="result-card" style="border-left-color: {severity_color};">
            <h2>✅ {disease_data['plant']} - Healthy!</h2>
            <p style="font-size: 1.1rem; color: #555;">{disease_data['description']}</p>
            <span class="severity-badge" style="background: {severity_color};">
                No Disease Detected
            </span>
            <p style="margin-top: 0.5rem; color: #777;">
                Confidence: <strong>{top_confidence*100:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-card" style="border-left-color: {severity_color};">
            <h2>⚠️ {disease_data['disease']}</h2>
            <p><strong>Plant:</strong> {disease_data['plant']}</p>
            <p style="font-size: 1.05rem; color: #555;">{disease_data['description']}</p>
            <span class="severity-badge" style="background: {severity_color};">
                Severity: {severity}
            </span>
            <p style="margin-top: 0.5rem; color: #777;">
                Confidence: <strong>{top_confidence*100:.1f}%</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # ── Top 3 Predictions ──
    st.markdown("### 📊 Top 3 Predictions")
    
    colors = ["#2d6a4f", "#52b788", "#95d5b2"]
    for i, (class_name, conf) in enumerate(results):
        info = get_disease_info(class_name)
        label = f"{info['plant']} → {info['disease']}" if info else class_name
        
        col1, col2 = st.columns([3, 7])
        with col1:
            medal = ["🥇", "🥈", "🥉"][i]
            st.markdown(f"**{medal} {label}**")
        with col2:
            st.markdown(render_confidence_bar(conf, colors[i]), unsafe_allow_html=True)
    
    # ── Detailed Info Tabs ──
    st.markdown("---")
    tab1, tab2, tab3 = st.tabs(["🔍 Symptoms", "💊 Treatment & Remedies", "📋 Full Report"])
    
    with tab1:
        st.markdown("#### Disease Symptoms")
        if is_healthy:
            st.success("No disease symptoms detected. Your plant looks healthy! 🌱")
        else:
            for symptom in disease_data["symptoms"]:
                st.markdown(f"""
                <div class="symptom-item">
                    🔸 {symptom}
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("#### Recommended Treatment")
        if is_healthy:
            st.info("🌱 **Preventive Care Tips:**")
        for treatment in disease_data["treatment"]:
            st.markdown(f"""
            <div class="treatment-item">
                💚 {treatment}
            </div>
            """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("#### Complete Diagnosis Report")
        
        report_data = {
            "Plant": disease_data["plant"],
            "Detected Disease": disease_data["disease"],
            "Severity Level": severity,
            "Confidence Score": f"{top_confidence*100:.2f}%",
            "Description": disease_data["description"]
        }
        
        for key, value in report_data.items():
            st.markdown(f"**{key}:** {value}")
        
        st.markdown("---")
        st.markdown("**All Predictions:**")
        for class_name, conf in results:
            info = get_disease_info(class_name)
            label = f"{info['plant']} - {info['disease']}" if info else class_name
            st.markdown(f"- {label}: `{conf*100:.2f}%`")


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 About")
    st.markdown("""
    **Plant Disease Detector** uses deep learning to identify 
    diseases in plant leaves from images.
    
    **How it works:**
    1. Upload a leaf image
    2. Our AI model analyzes it
    3. Get instant diagnosis & treatment
    
    ---
    
    **Model Details:**
    - Architecture: MobileNetV2 (Transfer Learning)
    - Dataset: PlantVillage (54,000+ images)
    - Classes: 38 (14 plants, 26 diseases)
    - Framework: TensorFlow / Keras
    
    ---
    
    **Supported Plants:**
    """)
    
    # Get unique plant names
    plants = sorted(set(info["plant"] for info in DISEASE_INFO.values()))
    for plant in plants:
        st.markdown(f"🌱 {plant}")
    
    st.markdown("---")
    st.markdown("""
    **⚠️ Disclaimer:**  
    This tool is for educational purposes. 
    Always consult an agricultural expert for 
    critical decisions.
    """)


# ──────────────────────────────────────────────
# Main Content
# ──────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🌿 Plant Disease Detector</h1>
    <p>Upload a leaf image to detect diseases instantly using AI</p>
</div>
""", unsafe_allow_html=True)

# Load model
model = load_model()

if model is None:
    st.warning("""
    ⚠️ **Model not found!** Please train the model first by running:
    ```bash
    python train.py --data_dir ./data/PlantVillage --epochs 15
    ```
    
    Or download a pre-trained model and place it in `models/plant_disease_model.keras`.
    
    ---
    
    **🎮 Demo Mode:** You can still explore the app interface below. 
    Upload an image to see how the UI works (predictions will be simulated).
    """)


# Upload Section
col_upload, col_preview = st.columns([1, 1])

with col_upload:
    st.markdown("### 📤 Upload Leaf Image")
    uploaded_file = st.file_uploader(
        "Choose an image of a plant leaf",
        type=["jpg", "jpeg", "png", "webp"],
        help="Upload a clear image of a single plant leaf for best results."
    )
    
    # Sample images info
    st.markdown("""
    <div class="info-section">
        <strong>📸 Tips for best results:</strong><br>
        • Use a clear, well-lit image<br>
        • Focus on a single leaf<br>
        • Avoid blurry or dark images<br>
        • Include visible symptoms if present
    </div>
    """, unsafe_allow_html=True)

with col_preview:
    if uploaded_file is not None:
        st.markdown("### 🖼️ Image Preview")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Uploaded Leaf Image")

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    if model is not None:
        with st.spinner("🔍 Analyzing leaf image..."):
            results = predict(model, image)
            display_results(results)
    else:
        # Demo mode with simulated results
        st.info("🎮 **Demo Mode** - Showing simulated results since no model is loaded.")
        with st.spinner("🔍 Simulating analysis..."):
            import time
            time.sleep(1)
            
            # Simulated results for demo
            demo_results = [
                ("Tomato___Early_blight", 0.87),
                ("Tomato___Late_blight", 0.08),
                ("Tomato___Septoria_leaf_spot", 0.03)
            ]
            display_results(demo_results)

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>
        🌿 <strong>Plant Disease Detector</strong> | 
        Built with TensorFlow & Streamlit | 
        Dataset: PlantVillage
    </p>
    <p>For educational purposes only. Not a substitute for professional agricultural advice.</p>
</div>
""", unsafe_allow_html=True)

# 🌿 Plant Disease Detector

An AI-powered web application that detects plant diseases from leaf images using deep learning. Built with **TensorFlow/Keras** and **Streamlit**.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 🎯 Features

- **Image Upload & Prediction** — Upload a leaf photo and get instant disease identification
- **Top-3 Predictions with Confidence Scores** — See the model's top 3 guesses with visual confidence bars
- **Disease Information** — Detailed descriptions of each detected disease
- **Symptoms Checklist** — Know what to look for on your plants
- **Treatment Recommendations** — Actionable remedies and prevention tips
- **Severity Assessment** — Color-coded severity levels (None / Moderate / High / Critical)
- **38 Disease Classes** — Covers 14 crop species and 26 diseases

## 🌱 Supported Plants & Diseases

| Plant | Diseases Detected |
|-------|------------------|
| Apple | Apple Scab, Black Rot, Cedar Apple Rust |
| Cherry | Powdery Mildew |
| Corn | Gray Leaf Spot, Common Rust, Northern Leaf Blight |
| Grape | Black Rot, Esca (Black Measles), Leaf Blight |
| Orange | Huanglongbing (Citrus Greening) |
| Peach | Bacterial Spot |
| Bell Pepper | Bacterial Spot |
| Potato | Early Blight, Late Blight |
| Squash | Powdery Mildew |
| Strawberry | Leaf Scorch |
| Tomato | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus |
| Blueberry, Raspberry, Soybean | Healthy detection |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/plant-disease-detector.git
cd plant-disease-detector
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate          # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Dataset

Download the **PlantVillage** dataset from Kaggle:

- **Link:** [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Extract it into a `data/` folder:

```
plant-disease-detector/
├── data/
│   └── PlantVillage/
│       ├── Apple___Apple_scab/
│       ├── Apple___Black_rot/
│       ├── ...
│       └── Tomato___healthy/
```

### 5. Train the Model

```bash
python train.py --data_dir ./data/PlantVillage --epochs 15 --batch_size 32
```

**Training outputs** (saved in `models/`):
- `plant_disease_model.keras` — Full Keras model
- `plant_disease_model.tflite` — Optimized model for mobile
- `class_names.json` — Class label mapping
- `training_history.png` — Accuracy/loss curves
- `classification_report.json` — Per-class metrics

### 6. Run the App

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 🏗️ Project Structure

```
plant-disease-detector/
│
├── app.py                      # Streamlit web application
├── train.py                    # Model training script
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
│
├── utils/
│   ├── __init__.py
│   └── disease_info.py         # Disease database (38 classes)
│
├── models/                     # Trained models (generated)
│   ├── plant_disease_model.keras
│   ├── plant_disease_model.tflite
│   ├── class_names.json
│   └── training_history.png
│
└── data/                       # Dataset (not in git)
    └── PlantVillage/
```

---

## 🧠 Model Architecture

- **Base Model:** MobileNetV2 (pre-trained on ImageNet)
- **Transfer Learning:** Feature extraction + fine-tuning
- **Training Strategy:**
  1. Phase 1: Train only the top layers (feature extraction)
  2. Phase 2: Unfreeze top layers of MobileNetV2 for fine-tuning
- **Data Augmentation:** Rotation, flip, zoom, brightness, shear
- **Input Size:** 224×224×3
- **Expected Accuracy:** ~95-97% on validation set

---

## ☁️ Deployment

### Deploy on Streamlit Cloud (Free & Easy)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. **Important:** Upload your trained model to the repo or use Git LFS

### Deploy on Render

1. Add a `Procfile`:
   ```
   web: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
   ```
2. Push to GitHub and connect to [render.com](https://render.com)

### Deploy on Hugging Face Spaces

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose "Streamlit" as the SDK
3. Upload your files including the model

---

## 📈 Resume Description

Here's how you can describe this project on your resume:

> **Plant Disease Detector** | Python, TensorFlow, Streamlit  
> - Built an AI-powered web app that identifies 26 plant diseases across 14 crop species from leaf images with 96%+ accuracy  
> - Implemented transfer learning using MobileNetV2 with a two-phase training strategy (feature extraction + fine-tuning)  
> - Designed an interactive UI with real-time predictions, confidence scores, severity assessment, and treatment recommendations  
> - Deployed as a full-stack web application using Streamlit, with TFLite model conversion for mobile readiness  

---

## 📝 License

This project is open source under the [MIT License](LICENSE).

---

## 🙏 Acknowledgments

- **Dataset:** [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) by Abdallah Ali
- **Base Model:** [MobileNetV2](https://arxiv.org/abs/1801.04381) by Google
- **Framework:** [TensorFlow](https://www.tensorflow.org/) & [Streamlit](https://streamlit.io/)

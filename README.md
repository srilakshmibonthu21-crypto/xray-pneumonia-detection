# A Deep CNN with Transfer Learning (VGG16) for X-Ray Pneumonia Detection

**Author:** BONTHU SRI LAKSHMI  
**Project Guide:** Mr. R. SriRamu  
**Domain:** Deep Learning & Computer Vision  
**Target:** Radiologist Second-Opinion Clinical Tool  

---

## 📌 Project Overview
Pneumonia is an acute inflammatory condition of the lung primarily affecting the alveoli. This project develops a deep Convolutional Neural Network (CNN) with transfer learning utilizing the **VGG16** architecture to classify chest X-ray radiographs into **Normal** or **Pneumonia**.

Data from the **ChestX-ray14** radiography dataset is cleaned, transformed, and augmented to train and evaluate the model. The final outcome is integrated with **Grad-CAM (Gradient-weighted Class Activation Mapping)** and delivered through an interactive **Streamlit Web Application** serving as an AI second-opinion assistant for radiologists.

---

## 🏗️ Repository Architecture
```
xray_pneumonia_detection/
├── README.md                      # Project setup & execution guide
├── requirements.txt               # Python package dependencies
├── app.py                         # Streamlit Radiologist Second-Opinion Web App
├── create_samples.py              # Generates sample synthetic X-rays for quick testing
├── docs/
│   └── PROJECT_SYNOPSIS.md        # Academic report, methodology, and viva Q&A
├── data/                          # Dataset directory (train / val / test)
│   ├── train/ (NORMAL, PNEUMONIA)
│   ├── val/   (NORMAL, PNEUMONIA)
│   └── test/  (NORMAL, PNEUMONIA)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Hyperparameters, image dimensions, paths
│   ├── dataset.py                 # Augmentation, data loaders, class balancing
│   ├── model.py                   # VGG16 Transfer Learning & Fine-Tuning layers
│   ├── train.py                   # 2-stage training script with early stopping & curves
│   ├── evaluate.py                # Sensitivity, Specificity, ROC-AUC, and Confusion Matrix
│   └── explainability.py          # Grad-CAM heatmap generator for clinical interpretability
└── results/                       # Generated curves, confusion matrices, metrics
```

---

## ⚙️ Quickstart Setup Guide

### 1. Create and Activate a Virtual Environment
```bash
python -m venv venv

# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Dataset Setup (ChestX-ray14 / Kaggle Chest X-ray)
Place the dataset in the `data/` folder following this structure:
```
data/
  ├── train/
  │     ├── NORMAL/
  │     └── PNEUMONIA/
  ├── val/
  │     ├── NORMAL/
  │     └── PNEUMONIA/
  └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```
*(If you wish to test the pipeline immediately without external downloads, run `python create_samples.py` to create mock sample X-rays).*

---

## 🚀 Running the Project

### Step 1: Model Training & Fine-Tuning
Execute the two-stage transfer learning training:
```bash
python -m src.train
```
- **Phase 1:** Trains custom dense classification head with frozen VGG16 base.
- **Phase 2:** Fine-tunes upper convolutional layers (`Block 5`) at a lower learning rate.
- Automatically saves `saved_models/vgg16_pneumonia_best.h5` and plots training curves to `results/training_curves.png`.

### Step 2: Clinical Performance Evaluation
Evaluate on the test cohort:
```bash
python -m src.evaluate
```
Outputs:
- **Sensitivity (Recall)** (Minimizes false negatives in clinical triage)
- **Specificity**
- **ROC-AUC Score**
- Confusion Matrix Heatmap (`results/evaluation_report.png`)

### Step 3: Launch the Radiologist Second-Opinion Web App
Run the interactive Streamlit user interface:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser. Radiologists can:
- Upload any Chest X-ray image (JPEG/PNG).
- Inspect instant prediction (Normal vs. Pneumonia) and confidence score.
- View real-time **Grad-CAM heatmaps** revealing the exact lung regions triggering the model's prediction.
- Adjust the confidence threshold slider to optimize screening sensitivity.

---

## 🔬 Model & Methodology Details
- **Base Network:** VGG16 (Pretrained on ImageNet)
- **Input Resolution:** $224 \times 224 \times 3$
- **Loss Function:** Binary Cross-Entropy
- **Optimizer:** Adam ($\alpha = 10^{-4}$ for Phase 1, $\alpha = 10^{-5}$ for Phase 2)
- **Regularization:** Batch Normalization, Dropout ($p = 0.50$), Early Stopping
- **Explainability:** Grad-CAM saliency applied to layer `block5_conv3`

---

## 📜 Academic Attribution
- **Candidate:** BONTHU SRI LAKSHMI
- **Supervisor:** Mr. R. SriRamu
- **Domain:** Deep Learning & Computer Vision

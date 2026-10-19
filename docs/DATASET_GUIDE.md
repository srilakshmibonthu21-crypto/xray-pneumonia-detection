# Dataset Setup Guide for Pneumonia Detection

**Project:** A deep CNN with transfer learning (VGG16) for X-ray pneumonia detection  
**Author:** BONTHU SRI LAKSHMI | **Guide:** Mr. R. SriRamu  

---

## 1. Quick Example Dataset (Already Provided)
Sample clinical chest radiographs have been downloaded directly into your project's [`data/`](file:///C:/Users/srila/.gemini/antigravity/scratch/xray_pneumonia_detection/data) folder:

```
data/
├── sample/
│   ├── NORMAL/
│   │   ├── normal_sample_1.jpeg
│   │   ├── normal_sample_2.jpeg
│   │   └── normal_sample_3.jpeg
│   └── PNEUMONIA/
│       ├── pneumonia_sample_1.jpeg
│       ├── pneumonia_sample_2.jpeg
│       ├── pneumonia_sample_3.jpeg
│       └── pneumonia_sample_4.jpeg
├── test/
├── train/
└── val/
```

These are real anterior-posterior chest radiographs from pediatric patients (Guangzhou Women and Children's Medical Center) demonstrating:
- **Normal:** Clear lung fields without abnormal opacifications.
- **Pneumonia:** Visible lung consolidations, cloudy infiltrates, and bronchial markings.

You can immediately select and test these in your Streamlit web application dropdown!

---

## 2. Downloading the Full Official Datasets

### Option A: Kaggle Chest X-Ray Images (Pneumonia) [Recommended for Training]
- **Size:** ~1.15 GB (5,856 Images)
- **Classes:** 2 Categories (NORMAL, PNEUMONIA)
- **Link:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Direct Download via Kaggle CLI:**
  ```bash
  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
  ```
  Unzip the contents directly into `data/`.

### Option B: Mendeley Data (Original Source)
- **Citation:** Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images”, Mendeley Data, V3, doi: 10.17632/rsc2jh6twg.3
- **Link:** [Mendeley Data Repository](https://data.mendeley.com/datasets/rsc2jh6twg/3)

### Option C: NIH ChestX-ray14 Dataset
- **Size:** 112,120 frontal-view X-ray images of 30,805 unique patients
- **Source:** National Institutes of Health (NIH) Clinical Center
- **Link:** [NIH Clinical Center ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- **Kaggle Mirror:** [NIH Chest X-rays on Kaggle](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## 3. Dataset Preprocessing Pipeline
When `python -m src.train` runs, the data loader automatically:
1. Resizes all radiographs to standard $224 \times 224 \times 3$ resolution.
2. Applies VGG16 zero-centering normalization (`preprocess_input`).
3. Augments images via random rotation ($\pm 15^\circ$), horizontal shift ($10\%$), zoom ($15\%$), and horizontal flip.
4. Calculates inverse frequency class weights to balance Normal and Pneumonia cohorts.

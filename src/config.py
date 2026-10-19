"""
Configuration settings and hyperparameters for Chest X-Ray Pneumonia Detection.
Project: A deep CNN with transfer learning (VGG16) for X-ray pneumonia detection
Author: BONTHU SRI LAKSHMI
Supervisor: Mr. R. SriRamu
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "saved_models"

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset specifications
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# Image and Model Hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3  # VGG16 requires 3-channel input
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

BATCH_SIZE = 32
RANDOM_SEED = 42

# Training Phase Hyperparameters
INITIAL_EPOCHS = 15
INITIAL_LR = 1e-4

# Fine-Tuning Phase Hyperparameters
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-5

# Class Labels
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
NUM_CLASSES = 1  # Binary classification using Sigmoid (or 2 with Softmax)

# Model Persistence
BEST_MODEL_PATH = MODELS_DIR / "vgg16_pneumonia_best.h5"
FINAL_MODEL_PATH = MODELS_DIR / "vgg16_pneumonia_final.keras"

"""
Data preprocessing, augmentation, and loader pipeline for Chest X-Ray images.
Supports ChestX-ray14 / Kaggle Chest X-ray pneumonia dataset layout:
  data/
    train/
      NORMAL/
      PNEUMONIA/
    val/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/
"""

import os
import numpy as np
from src import config

def get_data_generators(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE, img_size=config.IMG_SIZE):
    """
    Creates robust training, validation, and testing ImageDataGenerators with augmentation
    and VGG16-specific preprocessing.
    """
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications.vgg16 import preprocess_input

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Training augmentation: rotation, zoom, shear, horizontal flip
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Validation and Test: only preprocessing (no augmentation)
    eval_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = None
    val_generator = None
    test_generator = None

    if os.path.exists(train_dir):
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="binary",
            classes=config.CLASS_NAMES,
            shuffle=True,
            seed=config.RANDOM_SEED
        )

    if os.path.exists(val_dir):
        val_generator = eval_datagen.flow_from_directory(
            val_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="binary",
            classes=config.CLASS_NAMES,
            shuffle=False
        )

    if os.path.exists(test_dir):
        test_generator = eval_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode="binary",
            classes=config.CLASS_NAMES,
            shuffle=False
        )

    return train_generator, val_generator, test_generator

def get_class_weights(generator):
    """
    Computes balanced class weights to address data imbalance between Normal and Pneumonia classes.
    """
    from sklearn.utils.class_weight import compute_class_weight

    if generator is None or generator.classes is None:
        return None
    classes = generator.classes
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(classes),
        y=classes
    )
    class_weight_dict = dict(enumerate(weights))
    print(f"[INFO] Computed Class Weights: {class_weight_dict}")
    return class_weight_dict

def create_synthetic_sample_data():
    """
    Creates dummy X-ray like grayscale images in train/val/test directories
    to verify that the pipeline runs smoothly out of the box.
    """
    from PIL import Image, ImageDraw
    import random

    for split in ["train", "val", "test"]:
        n_samples = 8 if split == "train" else 4
        for cls_name in config.CLASS_NAMES:
            folder = config.DATA_DIR / split / cls_name
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(n_samples):
                file_path = folder / f"sample_{cls_name.lower()}_{i+1}.jpeg"
                if not file_path.exists():
                    # Generate a synthetic chest cavity appearance
                    img = Image.new("RGB", config.IMG_SIZE, color=(30, 30, 30))
                    draw = ImageDraw.Draw(img)
                    # Draw approximate lung field ellipses
                    draw.ellipse([30, 40, 100, 180], fill=(70, 70, 70))
                    draw.ellipse([124, 40, 194, 180], fill=(70, 70, 70))
                    if cls_name == "PNEUMONIA":
                        # Add cloudy consolidations / opacities
                        for _ in range(6):
                            rx = random.randint(40, 90)
                            ry = random.randint(70, 150)
                            draw.ellipse([rx, ry, rx+25, ry+25], fill=(160, 160, 160))
                    img.save(file_path, "JPEG")
    print(f"[INFO] Synthetic sample dataset created at: {config.DATA_DIR}")

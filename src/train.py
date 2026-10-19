"""
Training script for VGG16 Chest X-ray Pneumonia Detection.
Executes two-stage transfer learning:
Stage 1: Train custom classification head with frozen VGG16 base.
Stage 2: Fine-tune upper VGG16 convolutional layers (Block 5).
Includes ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, and training curve plotting.
"""

import os
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from src import config
from src.dataset import get_data_generators, get_class_weights, create_synthetic_sample_data
from src.model import build_vgg16_model, setup_fine_tuning

def plot_training_history(history_phase1, history_phase2=None, save_path=config.RESULTS_DIR / "training_curves.png"):
    """
    Plots training and validation Accuracy, Loss, and AUC curves across phases.
    """
    acc = history_phase1.history.get("accuracy", [])
    val_acc = history_phase1.history.get("val_accuracy", [])
    loss = history_phase1.history.get("loss", [])
    val_loss = history_phase1.history.get("val_loss", [])
    auc = history_phase1.history.get("auc", [])
    val_auc = history_phase1.history.get("val_auc", [])

    if history_phase2:
        acc += history_phase2.history.get("accuracy", [])
        val_acc += history_phase2.history.get("val_accuracy", [])
        loss += history_phase2.history.get("loss", [])
        val_loss += history_phase2.history.get("val_loss", [])
        auc += history_phase2.history.get("auc", [])
        val_auc += history_phase2.history.get("val_auc", [])

    epochs_range = range(1, len(acc) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Accuracy Plot
    axes[0].plot(epochs_range, acc, label="Train Accuracy", color="blue", linewidth=2)
    axes[0].plot(epochs_range, val_acc, label="Val Accuracy", color="orange", linestyle="--", linewidth=2)
    if history_phase2:
        axes[0].axvline(x=len(history_phase1.history["accuracy"]), color="green", linestyle=":", label="Fine-Tuning Start")
    axes[0].set_title("Classification Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].grid(True)
    axes[0].legend()

    # Loss Plot
    axes[1].plot(epochs_range, loss, label="Train Loss", color="blue", linewidth=2)
    axes[1].plot(epochs_range, val_loss, label="Val Loss", color="orange", linestyle="--", linewidth=2)
    if history_phase2:
        axes[1].axvline(x=len(history_phase1.history["loss"]), color="green", linestyle=":", label="Fine-Tuning Start")
    axes[1].set_title("Binary Cross-Entropy Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].grid(True)
    axes[1].legend()

    # AUC Plot
    axes[2].plot(epochs_range, auc, label="Train AUC", color="blue", linewidth=2)
    axes[2].plot(epochs_range, val_auc, label="Val AUC", color="orange", linestyle="--", linewidth=2)
    if history_phase2:
        axes[2].axvline(x=len(history_phase1.history.get("auc", [])), color="green", linestyle=":", label="Fine-Tuning Start")
    axes[2].set_title("Area Under ROC (AUC)")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("AUC")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[INFO] Training curves saved successfully to: {save_path}")

def train():
    print("=" * 60)
    print("CHEST X-RAY PNEUMONIA DETECTION: VGG16 TRANSFER LEARNING")
    print("Author: BONTHU SRI LAKSHMI | Guide: Mr. R. SriRamu")
    print("=" * 60)

    # 1. Check or prepare data
    train_gen, val_gen, test_gen = get_data_generators()

    if train_gen is None or train_gen.samples == 0:
        print("[WARNING] Real dataset not detected in data/train. Generating synthetic sample dataset for testing...")
        create_synthetic_sample_data()
        train_gen, val_gen, test_gen = get_data_generators()

    print(f"[INFO] Train samples: {train_gen.samples if train_gen else 0}")
    print(f"[INFO] Val samples:   {val_gen.samples if val_gen else 0}")
    print(f"[INFO] Test samples:  {test_gen.samples if test_gen else 0}")

    class_weights = get_class_weights(train_gen)

    # 2. Build Model
    model, base_model = build_vgg16_model()
    model.summary()

    # 3. Define Callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(config.BEST_MODEL_PATH),
            monitor="val_auc" if val_gen else "auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor="val_loss" if val_gen else "loss",
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss" if val_gen else "loss",
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        CSVLogger(str(config.RESULTS_DIR / "training_log.csv"))
    ]

    # 4. Phase 1: Train classification head
    print("\n--- PHASE 1: Training Classification Head with Frozen VGG16 Base ---")
    steps_per_epoch = max(1, train_gen.samples // config.BATCH_SIZE)
    val_steps = max(1, val_gen.samples // config.BATCH_SIZE) if val_gen else None

    history_p1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=config.INITIAL_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 5. Phase 2: Fine-Tuning Block 5
    print("\n--- PHASE 2: Fine-Tuning VGG16 Upper Convolutional Layers (Block 5) ---")
    model = setup_fine_tuning(model, base_model, unfreeze_blocks=1, learning_rate=config.FINE_TUNE_LR)

    history_p2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=val_steps,
        epochs=config.FINE_TUNE_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks
    )

    # 6. Save final model and plot results
    model.save(str(config.FINAL_MODEL_PATH))
    plot_training_history(history_p1, history_p2)
    print("\n[SUCCESS] Model training and fine-tuning complete!")

if __name__ == "__main__":
    train()

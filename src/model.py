"""
Model architecture definition using VGG16 Transfer Learning.
Implements:
1. Feature extraction with frozen ImageNet weights.
2. Custom Dense Classification Head (GlobalAveragePooling2D -> Dense -> Dropout -> Sigmoid).
3. Fine-tuning capability for adapting upper convolution blocks to Chest Radiography features.
"""

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from src import config

def build_vgg16_model(input_shape=config.INPUT_SHAPE, learning_rate=config.INITIAL_LR):
    """
    Constructs the VGG16 Transfer Learning model.
    Base convolutional layers are pre-trained on ImageNet and initially frozen.
    """
    # 1. Base Pretrained VGG16 without dense top layers
    base_model = VGG16(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze base model parameters for initial feature extraction
    base_model.trainable = False

    # 2. Build Classification Head
    inputs = layers.Input(shape=input_shape, name="input_xray")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dropout(0.5, name="dropout_0.5")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="VGG16_Pneumonia_Detector")

    # 3. Compile with clinical evaluation metrics
    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics
    )

    return model, base_model

def setup_fine_tuning(model, base_model, unfreeze_blocks=1, learning_rate=config.FINE_TUNE_LR):
    """
    Unfreezes the top convolutional block(s) (e.g. block5_conv1, block5_conv2, block5_conv3)
    and recompiles the model with a smaller learning rate for domain adaptation.
    """
    base_model.trainable = True

    # Freeze all layers except the last block
    # VGG16 has blocks: block1, block2, block3, block4, block5
    target_prefix = "block5_" if unfreeze_blocks == 1 else ("block4_", "block5_")
    
    for layer in base_model.layers:
        if any(layer.name.startswith(p) for p in (target_prefix if isinstance(target_prefix, tuple) else [target_prefix])):
            layer.trainable = True
        else:
            layer.trainable = False

    trainable_count = sum(len(layer.weights) for layer in model.layers if layer.trainable)
    print(f"[INFO] Fine-tuning setup completed. Trainable layers active.")

    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metrics
    )

    return model

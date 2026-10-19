"""
Explainable AI (XAI) using Grad-CAM (Gradient-weighted Class Activation Mapping).
Generates heatmaps highlighting infected lung opacities and infiltrates
to provide transparent visual evidence for radiologists.
"""

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from src import config

def make_gradcam_heatmap(img_array, model, last_conv_layer_name="block5_conv3", pred_index=None):
    """
    Computes Grad-CAM heatmap for a given input image array and model.
    """
    # Find the VGG16 base layer inside the model
    vgg_base = None
    for layer in model.layers:
        if "vgg16" in layer.name.lower():
            vgg_base = layer
            break

    if vgg_base is not None:
        # The model wraps VGG16 as a sub-model
        grad_model = tf.keras.models.Model(
            inputs=[vgg_base.input],
            outputs=[vgg_base.get_layer(last_conv_layer_name).output, vgg_base.output]
        )
        
        with tf.GradientTape() as tape:
            conv_outputs, base_features = grad_model(img_array)
            # Reconstruct the forward pass through the top classification head
            x = model.get_layer("global_avg_pool")(base_features)
            x = model.get_layer("dense_256")(x)
            x = model.get_layer("batch_norm")(x)
            x = model.get_layer("dropout_0.5")(x)
            preds = model.get_layer("pneumonia_output")(x)
            
            if pred_index is None:
                class_channel = preds[:, 0]
            else:
                class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
    else:
        # Flat model architecture
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_outputs, preds = grad_model(img_array)
            class_channel = preds[:, 0] if pred_index is None else preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel by its gradient importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Apply ReLU to retain only positive influence
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

def overlay_heatmap(original_img_pil, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Overlays Grad-CAM heatmap onto the original Chest X-ray.
    """
    orig_np = np.array(original_img_pil)
    if orig_np.ndim == 2:
        orig_np = cv2.cvtColor(orig_np, cv2.COLOR_GRAY2RGB)
    elif orig_np.shape[2] == 4:
        orig_np = orig_np[:, :, :3]

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (orig_np.shape[1], orig_np.shape[0]))

    # Convert heatmap to RGB colors
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Superimpose image and heatmap
    superimposed_img = np.uint8(alpha * heatmap_color + (1 - alpha) * orig_np)
    return superimposed_img, heatmap_resized

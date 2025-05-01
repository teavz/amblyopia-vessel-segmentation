#!/usr/bin/env python3

import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from imageio import imread

from SA_UNet import SA_UNet

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DESIRED_SIZE = 592
WEIGHTS_FILE = "SA_UNet.h5"


def initialize_model(weight_path: str = WEIGHTS_FILE) -> tf.keras.Model:
    """
    Build the SA-UNet model and load weights if the file exists.

    Args:
        weight_path: Path to the .h5 file containing pretrained weights.

    Returns:
        A SA-UNet model instance with weights loaded (if found).
    """
    model = SA_UNet(
        input_size=(DESIRED_SIZE, DESIRED_SIZE, 3),
        start_neurons=16,
        lr=1e-3,
        keep_prob=0.82,
        block_size=7,
    )

    if os.path.isfile(weight_path):
        model.load_weights(weight_path)
        print("Weights loaded successfully from", weight_path)
    else:
        print("Weights file not found:", weight_path)

    return model


def predict(
    model: tf.keras.Model,
    image_path: str,
    apply_clahe: bool = False,
) -> np.ndarray:
    """
    Run the model on a single IR fundus image and return a binary mask.

    Args:
        model: A SA-UNet model.
        image_path: Path to the input IR image.
        apply_clahe: If True, apply CLAHE preprocessing to the grayscale version.

    Returns:
        A uint8 NumPy array of shape (DESIRED_SIZE, DESIRED_SIZE) with values {0,255}.
    """
    image = imread(image_path)

    if apply_clahe:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(gray)

    # If grayscale, convert to 3-channel RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # If there's an alpha channel, drop it
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Resize & normalize
    image_resized = cv2.resize(image, (DESIRED_SIZE, DESIRED_SIZE))
    image_norm = image_resized.astype("float32") / 255.0
    batch = image_norm[np.newaxis, ...]

    expected_shape = (1, DESIRED_SIZE, DESIRED_SIZE, 3)
    if batch.shape != expected_shape:
        raise ValueError(
            f"Expected input shape {expected_shape}, got {batch.shape}"
        )

    x_tensor = tf.convert_to_tensor(batch)
    raw_pred = model(x_tensor)
    mask = K.eval(raw_pred)[0]  # shape: (592,592,1) or (592,592)

    mask = (mask * 255).astype(np.uint8)
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    return binary


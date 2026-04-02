#!/usr/bin/env python3

import os

import cv2
import numpy as np
import tensorflow as tf
from imageio import imread

from SA_UNet import SA_UNet

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
DESIRED_SIZE = 512
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


def _square_pad(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    dim = max(h, w)
    pad_top = (dim - h) // 2
    pad_bottom = dim - h - pad_top
    pad_left = (dim - w) // 2
    pad_right = dim - w - pad_left
    return cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, borderType=cv2.BORDER_REFLECT_101)


def _percentile_normalize(img: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    x = img.astype(np.float32)
    p2 = np.percentile(x, p_low)
    p98 = np.percentile(x, p_high)
    denom = max(p98 - p2, 1e-6)
    x = np.clip((x - p2) / denom, 0.0, 1.0)
    return x


def predict(
    model: tf.keras.Model,
    image_path: str,
    apply_clahe: bool = False,
    threshold: float | None = 0.5,
    morph_open: int = 0,
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

    # Ensure 3 channels (RGB), drop alpha
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.shape[-1] == 4:
        image = image[:, :, :3]

    # Optional CLAHE on luminance
    if apply_clahe:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        image = np.stack([eq, eq, eq], axis=-1)

    # Training-consistent preprocessing: square pad -> resize to 512; percentile normalize
    image_sq = _square_pad(image)
    image_resized = cv2.resize(image_sq, (DESIRED_SIZE, DESIRED_SIZE), interpolation=cv2.INTER_AREA)
    image_norm = _percentile_normalize(image_resized)
    batch = image_norm[np.newaxis, ...].astype("float32")

    expected_shape = (1, DESIRED_SIZE, DESIRED_SIZE, 3)
    if batch.shape != expected_shape:
        raise ValueError(
            f"Expected input shape {expected_shape}, got {batch.shape}"
        )

    x_tensor = tf.convert_to_tensor(batch)
    raw_pred = model(x_tensor)
    # Keras 3 / TF eager: convert tensor to numpy directly
    prob = raw_pred.numpy()[0]
    if prob.ndim == 3 and prob.shape[-1] == 1:
        prob = prob[..., 0]

    if threshold is None:
        return (prob * 255.0).astype(np.uint8)

    thr_val = max(0.0, min(1.0, float(threshold)))
    binary = (prob >= thr_val).astype(np.uint8) * 255

    if morph_open and int(morph_open) > 0:
        k = int(morph_open)
        k = max(1, min(15, k))
        kernel = np.ones((k, k), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    return binary

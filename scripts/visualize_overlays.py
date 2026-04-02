#!/usr/bin/env python3
"""
Create quick visual overlays of SA-UNet segmentations on input images.

Features
- Single image or entire directory
- Saves three panels per image: original, prediction, and color overlay
- Adjustable overlay color and alpha

Examples
  Single image:
    python scripts/visualize_overlays.py \
      --image "amblyopia-vessel-segmentation/sample-images/NOR_001_OD.png" \
      --model "amblyopia-vessel-segmentation/models/SA_UNet.h5"

  Batch (all PNGs in a folder):
    python scripts/visualize_overlays.py \
      --dir "amblyopia-vessel-segmentation/Participant Data/KS_OS/TSLO" \
      --model "amblyopia-vessel-segmentation/models/SA_UNet_TSLO.weights.h5" \
      --pattern "*.png" --alpha 0.35
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Iterable, List

import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# Local inference utils
from sa_unet_inference import initialize_model, predict


def collect_images(root: Path, pattern: str) -> List[Path]:
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(root.rglob(pattern))


def make_overlay(original_rgb: np.ndarray, mask_binary: np.ndarray, color=(0, 255, 0), alpha: float = 0.35) -> np.ndarray:
    """Blend a binary mask onto the original image.

    Args:
        original_rgb: HxWx3 uint8
        mask_binary: HxW uint8 in {0,255}
        color: BGR tuple for overlay color
        alpha: overlay opacity [0..1]
    """
    if original_rgb.ndim == 2:
        original_rgb = cv2.cvtColor(original_rgb, cv2.COLOR_GRAY2RGB)
    # ensure uint8 3‑channel
    base = original_rgb.copy()
    if base.shape[-1] == 4:
        base = base[:, :, :3]
    if mask_binary.ndim == 3:
        mask_binary = mask_binary[..., 0]

    color_layer = np.zeros_like(base)
    # Use RGB consistently for arrays loaded via imageio
    if len(color) == 3:
        col = (int(color[0]), int(color[1]), int(color[2]))
    else:
        col = (0, 255, 0)
    color_layer[mask_binary > 127] = col

    overlay = cv2.addWeighted(color_layer, alpha, base, 1.0, 0.0)
    # Keep vessel color only where mask is present; elsewhere just base image
    out = base.copy()
    m2d = (mask_binary > 127)
    # Use 2D boolean mask to index HxW on a HxWx3 array
    out[m2d] = overlay[m2d]
    return out


def visualize_triptych(original: np.ndarray, pred_mask_resized: np.ndarray, overlay: np.ndarray, title: str = ""):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(pred_mask_resized, cmap="gray")
    plt.title("Prediction (mask)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def process_one(model_path: str, image_path: Path, out_dir: Path, show: bool, alpha: float, color: tuple):
    # Lazy model init per call will be slow; keep a singleton instead
    # Here, we initialize once and capture in closure via attribute on function
    if not hasattr(process_one, "_model"):
        try:
            process_one._model = initialize_model(model_path)
        except TypeError:
            process_one._model = initialize_model()
            process_one._model.load_weights(model_path)

    model = process_one._model

    pred_mask = predict(model, str(image_path), apply_clahe=False)  # 592x592 binary

    # Read original at native resolution
    original = imageio.imread(str(image_path))
    if original.ndim == 2:
        original_rgb = np.stack([original, original, original], axis=-1)
    else:
        original_rgb = original[:, :, :3]

    # Resize mask to original size (nearest to preserve binary)
    h, w = original_rgb.shape[:2]
    pred_resized = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = make_overlay(original_rgb, pred_resized, color=color, alpha=alpha)

    # Save outputs
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    imageio.imwrite(out_dir / f"{stem}_pred.png", pred_resized)
    imageio.imwrite(out_dir / f"{stem}_overlay.png", overlay)

    if show:
        visualize_triptych(original_rgb, pred_resized, overlay, title=stem)


def parse_args():
    p = argparse.ArgumentParser(description="Visualize SA-UNet predictions as overlays")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to a single image")
    g.add_argument("--dir", type=str, help="Directory to scan for images")
    p.add_argument("--pattern", type=str, default="*.png", help="Glob to match images under --dir")
    p.add_argument("--model", type=str, required=True, help="Path to model weights (.h5)")
    p.add_argument("--alpha", type=float, default=0.35, help="Overlay opacity [0..1]")
    p.add_argument("--color", type=str, default="0,255,0", help="Overlay RGB color, e.g. 0,255,0 for green")
    p.add_argument("--out", type=str, default="segmentation-overlays", help="Output folder for saved visuals")
    p.add_argument("--show", action="store_true", help="Display triptych window with matplotlib")
    return p.parse_args()


def parse_color(s: str) -> tuple:
    try:
        parts = [int(x) for x in s.split(",")]
        if len(parts) != 3:
            raise ValueError
        parts = [max(0, min(255, v)) for v in parts]
        return (parts[0], parts[1], parts[2])
    except Exception:
        return (0, 255, 0)


def main():
    args = parse_args()
    out_dir = Path(args.out)
    color = parse_color(args.color)

    targets: Iterable[Path]
    if args.image:
        targets = [Path(args.image)]
    else:
        targets = collect_images(Path(args.dir), args.pattern)

    if not targets:
        raise SystemExit("No input images found.")

    # Initialize model once before loop for speed
    try:
        model = initialize_model(args.model)
    except TypeError:
        model = initialize_model()
        model.load_weights(args.model)
    # bind to function so process_one can reuse
    process_one._model = model

    for img_path in targets:
        try:
            process_one(args.model, img_path, out_dir, args.show, args.alpha, color)
            print(f"Saved overlay for {img_path}")
        except Exception as e:
            print(f"[WARN] Failed on {img_path}: {e}")


if __name__ == "__main__":
    main()

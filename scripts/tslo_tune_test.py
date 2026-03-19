#!/usr/bin/env python3
"""
Evaluate a fine‑tuned SA‑UNet on the held‑out participant test split.

- Uses the same data discovery + preprocessing as finetune_tslo.py
- Reports continuous Dice/IoU (on probabilities) via model.evaluate
- Optionally reports thresholded Dice at a chosen probability threshold
  (reads <weights>.threshold.json if available, else defaults to 0.5 unless overridden)

Usage (example):
  python scripts/tslo_tune_test.py \
    --data-root "<Participant Data>" \
    --weights models/SA_UNet_TSLO.weights.h5 \
    --img-size 512 --batch-size 2 --use-clahe

To force a threshold (for thresholded Dice): add --threshold 0.5 (or other value)
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf

# Local imports: reuse dataset and metrics from training script
SCRIPT_DIR = Path(__file__).resolve().parent
import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from SA_UNet import SA_UNet  # noqa: E402
from finetune_tslo import (
    build_samples,
    participant_level_split,
    make_dataset,
    dice_coefficient,
    iou_metric,
)


def load_threshold_from_sidecar(weights_path: str) -> float | None:
    """Return threshold from <weights>.threshold.json if it exists, else None."""
    base = Path(weights_path)
    # finetune_tslo wrote threshold using .with_suffix('') + '.threshold.json'
    # If weights end with .weights.h5, strip that; otherwise still try suffix replacement
    if base.suffix == '.h5' and base.name.endswith('.weights.h5'):
        thr_json = base.with_suffix('').as_posix() + '.threshold.json'
    else:
        thr_json = base.with_suffix('.threshold.json').as_posix()
    if os.path.isfile(thr_json):
        try:
            with open(thr_json, 'r') as f:
                js = json.load(f)
            t = js.get('best_threshold', None)
            return float(t) if t is not None else None
        except Exception:
            return None
    return None


def dice_at_threshold(model: tf.keras.Model, ds: tf.data.Dataset, thr: float) -> float:
    inter = 0.0
    sum_pred = 0.0
    sum_true = 0.0
    for imgs, masks in ds:
        preds = model.predict(imgs, verbose=0)
        pm = (preds >= thr).astype(np.float32)
        masks = masks.numpy().astype(np.float32)
        inter += float((pm * masks).sum())
        sum_pred += float(pm.sum())
        sum_true += float(masks.sum())
    return (2.0 * inter + 1.0) / (sum_pred + sum_true + 1.0)


def parse_args():
    ap = argparse.ArgumentParser(description='Evaluate TSLO model on held-out participants')
    ap.add_argument('--data-root', required=True, type=str)
    ap.add_argument('--weights', required=True, type=str)
    ap.add_argument('--img-size', type=int, default=512)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--train-ratio', type=float, default=0.70)
    ap.add_argument('--val-ratio', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--use-clahe', action='store_true')
    ap.add_argument('--threshold', type=float, default=-1.0, help='If <0, read sidecar or use 0.5; else use given value')
    return ap.parse_args()


def main():
    args = parse_args()

    # Seed for reproducibility
    import random as _py_random
    _py_random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_root = Path(args.data_root)
    assert data_root.exists(), f"Data root not found: {data_root}"

    samples = build_samples(data_root)
    train_s, val_s, test_s = participant_level_split(samples, args.train_ratio, args.val_ratio, seed=args.seed)
    if not test_s:
        print('No test participants in the split; consider changing --seed or ratios')

    ds_test = make_dataset(test_s, args.img_size, args.batch_size, shuffle=False, augment=False, use_clahe=args.use_clahe)

    # Build and load model
    model = SA_UNet(input_size=(args.img_size, args.img_size, 3), start_neurons=16, lr=1e-3, keep_prob=0.82, block_size=7)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy',
                  metrics=[dice_coefficient, iou_metric])
    model.load_weights(args.weights)

    print("Evaluating continuous metrics (probability maps, no threshold)...")
    metrics = model.evaluate(ds_test, verbose=1)
    metrics_dict = dict(zip(model.metrics_names, metrics))
    print({k: float(v) for k, v in metrics_dict.items()})

    # Thresholded Dice (optional)
    thr = None
    if args.threshold is not None and args.threshold >= 0:
        thr = float(args.threshold)
    else:
        thr = load_threshold_from_sidecar(args.weights)
        if thr is None:
            thr = 0.5
    print(f"Computing thresholded Dice at threshold={thr:.3f} ...")
    dthr = dice_at_threshold(model, ds_test, thr)
    print({"dice_at_threshold": float(dthr), "threshold": float(thr)})


if __name__ == '__main__':
    main()

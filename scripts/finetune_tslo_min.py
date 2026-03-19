#!/usr/bin/env python3
"""
Minimal fine-tuning of OCT‑trained SA‑UNet to TSLO images.

Design goals: simplest possible pipeline to avoid adding noise.
- Data discovery: participant-level, simple mask name matching.
- I/O: imageio (requires imagecodecs for LZW); grayscale→RGB; mask binarize.
- Preprocessing: scale to [0,1], direct resize to square (no CLAHE, no percentile norm).
- No augmentations.
- Single-stage training: all layers trainable with a small LR.
- Callbacks: best-checkpoint + early stopping only.

Usage example:
  python scripts/finetune_tslo_min.py \
    --data-root "<Participant Data>" \
    --base-weights models/SA_UNet.h5 \
    --out-weights models/SA_UNet_TSLO.weights.h5
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

# Robust image I/O (LZW TIFF requires imagecodecs)
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
import sys
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from SA_UNet import SA_UNet  # noqa: E402

# ------------------------------- Discovery ---------------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
MASK_SUFFIXES = ("", "_mask", "-mask", ".mask", "_seg", "-seg", "_vessel", "-vessel")
MASK_DIR_CANDIDATES = ("masks", "labels", "annotations")


def _is_mask_like(p: Path) -> bool:
    s = p.stem.lower()
    return any(k in s for k in ("mask", "seg", "label", "vessel"))


def list_participants(data_root: Path) -> List[Path]:
    return [p for p in sorted(data_root.iterdir()) if p.is_dir()]


def gather_images(participant_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in IMG_EXTS:
        files.extend(participant_dir.rglob(f"*{ext}"))
    files = [f for f in files if not _is_mask_like(f)]
    return files


def find_mask_for_image(img: Path, participant_dir: Path) -> Optional[Path]:
    stem = img.stem
    candidates_dirs: List[Path] = []
    candidates_dirs.append(img.parent)
    for name in MASK_DIR_CANDIDATES:
        candidates_dirs.append(img.parent / name)
        candidates_dirs.append(participant_dir / name)
    # try combinations (no deep ancestor crawling to keep simple)
    for d in candidates_dirs:
        if not d.exists() or not d.is_dir():
            continue
        for suf in MASK_SUFFIXES:
            for ext in IMG_EXTS:
                cand = d / f"{stem}{suf}{ext}"
                if cand.exists():
                    return cand
        # fallback: same stem any ext
        for ext in IMG_EXTS:
            cand = d / f"{stem}{ext}"
            if cand.exists():
                return cand
    return None


def build_samples(data_root: Path, verbose: bool = True) -> List[Tuple[str, str, str]]:
    samples: List[Tuple[str, str, str]] = []
    for participant_dir in list_participants(data_root):
        imgs = gather_images(participant_dir)
        found = 0
        for img in imgs:
            mask = find_mask_for_image(img, participant_dir)
            if mask is None:
                continue
            samples.append((participant_dir.name, str(img), str(mask)))
            found += 1
        if verbose:
            print(f"Participant {participant_dir.name}: {found} paired images")
    if verbose:
        print(f"Total paired samples: {len(samples)}")
    return samples


def participant_level_split(samples: Sequence[Tuple[str, str, str]],
                            train_ratio: float = 0.7,
                            val_ratio: float = 0.15,
                            seed: int = 42):
    import random
    rng = random.Random(seed)
    parts = sorted({s[0] for s in samples})
    rng.shuffle(parts)
    n = len(parts)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    train_parts = set(parts[:n_train])
    val_parts = set(parts[n_train:n_train + n_val])
    test_parts = set(parts[n_train + n_val:])

    def filt(which):
        return [s for s in samples if s[0] in which]

    train = filt(train_parts)
    val = filt(val_parts)
    test = filt(test_parts)
    print(f"Participants split: train={len(train_parts)}, val={len(val_parts)}, test={len(test_parts)}")
    print(f"Samples split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


# ------------------------------ Image I/O ----------------------------------
def _ensure_imageio():
    if imageio is None:
        raise SystemExit("imageio is required; install imageio and imagecodecs for TIFF LZW support")


def _numpy_load(img_path: str, mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_imageio()
    img = imageio.imread(img_path)
    msk = imageio.imread(mask_path)
    # channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if msk.ndim == 3:
        msk = msk[..., 0]
    # Convert to float and apply per-image percentile normalization (p2–p98)
    img = img.astype(np.float32)
    p2 = float(np.percentile(img, 2))
    p98 = float(np.percentile(img, 98))
    denom = max(p98 - p2, 1e-6)
    img = np.clip((img - p2) / denom, 0.0, 1.0)
    msk = (msk > 127).astype(np.float32)  # {0,1}
    return img, msk


def resize_square_pad(x: tf.Tensor, size: int, method=tf.image.ResizeMethod.BILINEAR) -> tf.Tensor:
    """Reflect-pad to square then resize (preserve aspect)."""
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    dim = tf.maximum(h, w)
    pad_h = dim - h
    pad_w = dim - w
    x = tf.pad(x, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]], mode='REFLECT')
    x = tf.image.resize(x, (size, size), method=method)
    return x


def make_dataset(samples: Sequence[Tuple[str, str, str]],
                 img_size: int,
                 batch_size: int,
                 shuffle: bool,
                 cache: bool = True) -> tf.data.Dataset:
    paths_imgs = [s[1] for s in samples]
    paths_masks = [s[2] for s in samples]

    def _load(img_path, mask_path):
        def _py(ip, mp):
            ip = ip.decode() if isinstance(ip, (bytes, bytearray)) else str(ip)
            mp = mp.decode() if isinstance(mp, (bytes, bytearray)) else str(mp)
            img, m = _numpy_load(ip, mp)
            return img.astype(np.float32), m.astype(np.float32)
        img, msk = tf.numpy_function(_py, [img_path, mask_path], [tf.float32, tf.float32])
        img.set_shape([None, None, 3])
        msk.set_shape([None, None])
        # reflect pad to square, then resize (aspect-preserving)
        img = resize_square_pad(img, img_size, method=tf.image.ResizeMethod.BILINEAR)
        msk = tf.expand_dims(msk, -1)
        msk = resize_square_pad(msk, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # ensure binary {0,1}
        msk = tf.cast(msk > 0.5, tf.float32)
        return img, msk

    ds = tf.data.Dataset.from_tensor_slices((paths_imgs, paths_masks))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(paths_imgs), reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------- Losses & Metrics -----------------------------
def dice_coefficient(y_true, y_pred, smooth: float = 1.0):
    y_true_f = tf.reshape(y_true, (-1,))
    y_pred_f = tf.reshape(y_pred, (-1,))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denom + smooth)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - dice_coefficient(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice


# ------------------------------- Training ----------------------------------
def build_model(img_size: int, base_weights: str, lr: float) -> tf.keras.Model:
    model = SA_UNet(input_size=(img_size, img_size, 3), start_neurons=16, lr=1e-3, keep_prob=0.82, block_size=7)
    if base_weights and os.path.isfile(base_weights):
        try:
            model.load_weights(base_weights)
            print(f"Loaded base weights: {base_weights}")
        except Exception as e:
            print(f"WARNING: failed to load base weights: {e}\nTraining from random init.")
    else:
        print("Base weights not provided or not found — training from scratch initial weights.")
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice_coefficient])
    return model


# --------------------------------- CLI -------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Minimal fine-tune of SA-UNet on TSLO images (single-stage, no augs).")
    p.add_argument('--data-root', type=str, required=True, help='Root folder with participant subfolders.')
    p.add_argument('--base-weights', type=str, default=str(REPO_ROOT / 'models' / 'SA_UNet.h5'),
                   help='Path to initial SA-UNet weights (.h5).')
    p.add_argument('--out-weights', type=str, default=str(REPO_ROOT / 'models' / 'SA_UNet_TSLO.weights.h5'),
                   help='Where to save best fine-tuned weights (.weights.h5).')
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--lr', type=float, default=5e-5, help='Fixed learning rate (default 5e-5)')
    return p.parse_args()


def ensure_weights_suffix(path: str) -> str:
    if path.endswith('.weights.h5'):
        return path
    base, _ = os.path.splitext(path)
    adj = base + '.weights.h5'
    print(f"Adjusted output path to weights filename: {adj}")
    return adj


def train():
    args = parse_args()

    # Seed
    import random as _py_random
    _py_random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    data_root = Path(args.data_root).resolve()
    assert data_root.exists(), f"Data root not found: {data_root}"

    samples = build_samples(data_root)
    if len(samples) == 0:
        raise SystemExit("No (image, mask) pairs found. Verify mask filenames and locations.")

    # Minimal participant-level split (fixed 70/15/15)
    train_s, val_s, _ = participant_level_split(samples, 0.70, 0.15, seed=args.seed)

    ds_train = make_dataset(train_s, args.img_size, args.batch_size, shuffle=True)
    ds_val = make_dataset(val_s, args.img_size, args.batch_size, shuffle=False)

    model = build_model(args.img_size, args.base_weights, args.lr)

    out_weights = ensure_weights_suffix(args.out_weights)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=out_weights,
            monitor='val_dice_coefficient',
            mode='max',
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient', mode='max', patience=8, restore_best_weights=True, verbose=1
        ),
    ]

    print(f"Training: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")
    _ = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    print(f"Best weights saved to: {out_weights}")


if __name__ == '__main__':
    train()

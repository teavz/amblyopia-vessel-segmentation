#!/usr/bin/env python3
"""
Fine-tune an OCT-trained SA-UNet to TSLO images (training only).

- Participant-level splits to avoid leakage
- Robust TIFF reading via imageio (+ imagecodecs for LZW)
- TSLO-friendly preprocessing and augmentations
- Two-stage schedule: freeze -> unfreeze
- Defaults to 512x512 to match typical OCT-trained weights

Usage example:
  python scripts/finetune_tslo.py \
    --data-root "<Participant Data root>" \
    --base-weights models/SA_UNet.h5 \
    --out-weights models/SA_UNet_TSLO.weights.h5 \
    --epochs 40 --batch-size 4 --img-size 512
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

# Optional: CLAHE for contrast enhancement
try:
    from skimage import exposure as sk_exposure
except Exception:
    sk_exposure = None

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
    # ascend up to two levels
    parent = img.parent
    for _ in range(2):
        parent = parent.parent
        if parent is None:
            break
        for name in MASK_DIR_CANDIDATES:
            candidates_dirs.append(parent / name)
    # try combinations
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
        raise RuntimeError("imageio is required; install imageio and imagecodecs for TIFF LZW support")


def _apply_clahe(img: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    if sk_exposure is None:
        return img
    gray = img[..., 0].astype(np.float32) / 255.0
    eq = sk_exposure.equalize_adapthist(gray, clip_limit=clip_limit, kernel_size=tile_size)
    eq = (eq * 255.0).astype(np.uint8)
    out = np.stack([eq, eq, eq], axis=-1)
    return out


def _numpy_load_and_preprocess(img_path: str, mask_path: str, use_clahe: bool) -> Tuple[np.ndarray, np.ndarray]:
    _ensure_imageio()
    # read
    img = imageio.imread(img_path)
    msk = imageio.imread(mask_path)
    # channels
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    if msk.ndim == 3:
        msk = msk[..., 0]
    # optional CLAHE pre-normalization
    if use_clahe:
        img = _apply_clahe(img)
    # percentile normalization (p2-p98) per image
    img = img.astype(np.float32)
    p2 = np.percentile(img, 2)
    p98 = np.percentile(img, 98)
    denom = max(p98 - p2, 1e-6)
    img = np.clip((img - p2) / denom, 0.0, 1.0)
    # binarize mask
    msk = (msk > 127).astype(np.uint8)
    return img, msk


def make_dataset(samples: Sequence[Tuple[str, str, str]],
                 img_size: int,
                 batch_size: int,
                 shuffle: bool,
                 augment: bool,
                 use_clahe: bool = False,
                 cache: bool = True) -> tf.data.Dataset:
    paths_imgs = [s[1] for s in samples]
    paths_masks = [s[2] for s in samples]

    def _load(img_path, mask_path):
        def _py(ip, mp):
            ip = ip.decode() if isinstance(ip, (bytes, bytearray)) else str(ip)
            mp = mp.decode() if isinstance(mp, (bytes, bytearray)) else str(mp)
            img, m = _numpy_load_and_preprocess(ip, mp, use_clahe)
            return img.astype(np.float32), m.astype(np.uint8)
        img, msk = tf.numpy_function(_py, [img_path, mask_path], [tf.float32, tf.uint8])
        img.set_shape([None, None, 3])
        msk.set_shape([None, None])
        # pad to square, resize
        img = resize_square_pad(img, img_size)
        msk = tf.expand_dims(msk, -1)
        msk = resize_square_pad(msk, img_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # ensure mask is floating before threshold/augs
        msk = tf.cast(msk, tf.float32)
        # augment
        if augment:
            img, msk = augment_pair(img, msk)
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

# ------------------------------ Transforms ---------------------------------

def resize_square_pad(x: tf.Tensor, size: int, method=tf.image.ResizeMethod.BILINEAR) -> tf.Tensor:
    h = tf.shape(x)[0]
    w = tf.shape(x)[1]
    dim = tf.maximum(h, w)
    pad_h = dim - h
    pad_w = dim - w
    x = tf.pad(x, [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]], mode='REFLECT')
    x = tf.image.resize(x, (size, size), method=method)
    return x


def augment_pair(img: tf.Tensor, msk: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    # flips
    flip_lr = tf.random.uniform(()) > 0.5
    flip_ud = tf.random.uniform(()) > 0.5
    img, msk = tf.cond(flip_lr,
        lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(msk)),
        lambda: (img, msk))
    img, msk = tf.cond(flip_ud,
        lambda: (tf.image.flip_up_down(img), tf.image.flip_up_down(msk)),
        lambda: (img, msk))
    # rotations
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k)
    msk = tf.image.rot90(msk, k)
    # brightness/contrast
    img = tf.image.random_brightness(img, max_delta=0.05)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    # gamma
    gamma = tf.random.uniform((), 0.9, 1.1)
    img = tf.pow(tf.clip_by_value(img, 0.0, 1.0), gamma)
    # Gaussian noise
    noise = tf.random.normal(tf.shape(img), mean=0.0, stddev=0.02, dtype=img.dtype)
    img = tf.clip_by_value(img + noise, 0.0, 1.0)
    return img, msk

# ---------------------------- Losses & Metrics -----------------------------

def dice_coefficient(y_true, y_pred, smooth: float = 1.0):
    y_true_f = tf.reshape(y_true, (-1,))
    y_pred_f = tf.reshape(y_pred, (-1,))
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denom = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    return (2.0 * intersection + smooth) / (denom + smooth)


def iou_metric(y_true, y_pred, eps: float = 1e-6):
    y_true_f = tf.reshape(y_true, (-1,))
    y_pred_f = tf.reshape(y_pred, (-1,))
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - inter
    return (inter + eps) / (union + eps)


def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = 1.0 - dice_coefficient(y_true, y_pred)
    return 0.5 * bce + 0.5 * dice

# ------------------------------- Training ----------------------------------

def build_model(img_size: int, base_weights: str, base_lr: float, clipnorm: float = 0.0) -> tf.keras.Model:
    model = SA_UNet(input_size=(img_size, img_size, 3), start_neurons=16, lr=1e-3, keep_prob=0.82, block_size=7)
    if base_weights and os.path.isfile(base_weights):
        try:
            model.load_weights(base_weights)
            print(f"Loaded base weights: {base_weights}")
        except Exception as e:
            print(f"WARNING: failed to load base weights: {e}\nTraining from random init.")
    else:
        print("Base weights not provided or not found — training from scratch initial weights.")
    if clipnorm and clipnorm > 0:
        opt = tf.keras.optimizers.Adam(learning_rate=base_lr, clipnorm=float(clipnorm))
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=base_lr)
    model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[dice_coefficient, iou_metric])
    return model


def freeze_early_layers(model: tf.keras.Model, freeze_fraction: float = 0.4):
    total = len(model.layers)
    cutoff = int(total * freeze_fraction)
    for i, layer in enumerate(model.layers):
        layer.trainable = (i >= cutoff)
    print(f"Froze first {cutoff}/{total} layers (~{int(freeze_fraction*100)}%).")


def unfreeze_all(model: tf.keras.Model):
    for layer in model.layers:
        layer.trainable = True
    print("Unfroze all layers.")


def ensure_weights_suffix(path: str) -> str:
    if path.endswith('.weights.h5'):
        return path
    base, _ = os.path.splitext(path)
    adj = base + '.weights.h5'
    print(f"Adjusted output path to weights filename: {adj}")
    return adj


def maybe_set_mixed_precision(enable: bool):
    if not enable:
        return
    try:
        from tensorflow.keras.mixed_precision import experimental as mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
        print("Enabled mixed precision (experimental API).")
    except Exception:
        try:
            from tensorflow import keras
            keras.mixed_precision.set_global_policy('mixed_float16')
            print("Enabled mixed precision (global policy).")
        except Exception as e:
            print(f"Mixed precision not available: {e}")

# --------------------------------- CLI -------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SA-UNet on TSLO images with participant-level splits.")
    p.add_argument('--data-root', type=str, required=True, help='Root folder containing participant subfolders.')
    p.add_argument('--base-weights', type=str, default=str(REPO_ROOT / 'models' / 'SA_UNet.h5'),
                   help='Path to initial SA-UNet weights (.h5).')
    p.add_argument('--out-weights', type=str, default=str(REPO_ROOT / 'models' / 'SA_UNet_TSLO.weights.h5'),
                   help='Where to save best fine-tuned weights (.h5).')
    p.add_argument('--img-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=0, help='Batch size (0=auto: GPU:4, CPU:2)')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument('--val-ratio', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use-clahe', action='store_true')
    p.add_argument('--mixed-precision', action='store_true')
    p.add_argument('--tensorboard', type=str, default='')
    p.add_argument('--warmup-epochs', type=int, default=8)
    p.add_argument('--freeze-fraction', type=float, default=0.4)
    # New tuning knobs
    p.add_argument('--lr-stage1', type=float, default=0.0, help='Warmup LR; 0=auto scale (1e-4 * bs/4)')
    p.add_argument('--lr-stage2', type=float, default=0.0, help='Unfreeze LR; 0=auto scale (5e-5 * bs/4)')
    p.add_argument('--patience-es', type=int, default=10, help='EarlyStopping patience (epochs)')
    p.add_argument('--patience-rlr', type=int, default=5, help='ReduceLROnPlateau patience (epochs)')
    p.add_argument('--clipnorm', type=float, default=0.0, help='Gradient clipnorm (0=disabled)')
    return p.parse_args()


def probe_tiff_readability(samples: Sequence[Tuple[str, str, str]], max_checks: int = 8):
    failures = {}
    if imageio is None:
        failures['imageio'] = 'imageio not importable'
        return failures
    checked = 0
    for _, ip, _ in samples:
        if not (ip.lower().endswith('.tif') or ip.lower().endswith('.tiff')):
            continue
        try:
            _ = imageio.imread(ip)
        except Exception as e:
            failures[ip] = str(e)
        checked += 1
        if checked >= max_checks:
            break
    return failures


def train():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    assert data_root.exists(), f"Data root not found: {data_root}"

    # Reproducibility
    import random as _py_random
    _py_random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    samples = build_samples(data_root)
    if len(samples) == 0:
        raise SystemExit("No (image, mask) pairs found. Verify mask filenames and locations.")

    # preflight TIFF codec
    failures = probe_tiff_readability(samples)
    if failures:
        print("\nRead errors detected for some TIFFs:")
        for p, err in failures.items():
            print(f" - {p}: {err}")
        raise SystemExit("Missing codecs for TIFF? Install imagecodecs and retry.")

    train_s, val_s, test_s = participant_level_split(samples, args.train_ratio, args.val_ratio, seed=args.seed)

    # Auto-select batch size if requested
    bs = args.batch_size
    if bs is None or bs <= 0:
        has_gpu = len(tf.config.list_physical_devices('GPU')) > 0
        bs = 4 if has_gpu else 2
        print(f"Auto-selected batch size: {bs} ({'GPU' if has_gpu else 'CPU'} detected)")

    ds_train = make_dataset(train_s, args.img_size, bs, shuffle=True, augment=True, use_clahe=args.use_clahe)
    ds_val = make_dataset(val_s, args.img_size, bs, shuffle=False, augment=False, use_clahe=args.use_clahe)

    maybe_set_mixed_precision(args.mixed_precision)

    # Learning rates (auto-scale by batch size unless explicitly provided)
    base_lr_stage1 = (args.lr_stage1 if args.lr_stage1 and args.lr_stage1 > 0 else (1e-4 * (bs / 4.0)))
    lr_stage2 = (args.lr_stage2 if args.lr_stage2 and args.lr_stage2 > 0 else (5e-5 * (bs / 4.0)))
    print(f"LRs: stage1={base_lr_stage1:.6g}, stage2={lr_stage2:.6g}; clipnorm={args.clipnorm}")

    model = build_model(args.img_size, args.base_weights, base_lr_stage1, clipnorm=args.clipnorm)

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
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_dice_coefficient', mode='max', factor=0.5, patience=int(args.patience_rlr), min_lr=1e-6, verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_dice_coefficient', mode='max', patience=int(args.patience_es), restore_best_weights=True, verbose=1
        ),
    ]
    if args.tensorboard:
        callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=args.tensorboard, histogram_freq=0))

    # Stage 1: warmup with frozen early layers
    freeze_early_layers(model, freeze_fraction=args.freeze_fraction)
    _ = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=max(1, args.warmup_epochs),
        verbose=1,
        callbacks=callbacks,
    )

    # Stage 2: unfreeze all
    unfreeze_all(model)
    # Adjust LR for stage 2 (possibly user-specified)
    try:
        model.optimizer.learning_rate.assign(lr_stage2)
    except Exception:
        tf.keras.backend.set_value(model.optimizer.learning_rate, lr_stage2)
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

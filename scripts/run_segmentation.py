#!/usr/bin/env python3

import argparse
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import tensorflow as tf

from sa_unet_inference import initialize_model, predict


def visualize_results(image_path: str, segmented_image):
    """Display the original and segmented IR fundus images."""
    raw = tf.io.read_file(image_path)
    original = tf.image.decode_png(raw, channels=3)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original.numpy())
    plt.title("Original IR Fundus Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image, cmap="gray")
    plt.title("Segmented Vessel Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run SA-UNet segmentation on an IR fundus image."
    )
    parser.add_argument(
        "--image", "-i", required=True,
        help="Path to the input IR fundus image (e.g. .png)"
    )
    parser.add_argument(
        "--model", "-m", required=True,
        help="Path to the trained SA-UNet model file (e.g. .h5)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # load model architecture + weights
    # if initialize_model can take a path, pass it in;
    # otherwise load weights afterward.
    try:
        model = initialize_model(args.model)
    except TypeError:
        # fallback if initialize_model() takes no args
        model = initialize_model()
        model.load_weights(args.model)

    # run prediction
    segmentation = predict(model, args.image, apply_clahe=False)
    
    # Upsample to original size
    orig = imageio.imread(args.image)
    orig_h, orig_w = orig.shape[:2]
    segmentation_original = cv2.resize(
       segmentation,
       (orig_w, orig_h),
       interpolation=cv2.INTER_NEAREST
   )
    
    # Save segmented image
    output_dir = "segmented-images"
    os.makedirs(output_dir, exist_ok=True)
    
    stem, _ = os.path.splitext(os.path.basename(args.image))
    save_path = os.path.join(output_dir, f"{stem}_segmented.png")
    
    imageio.imwrite(save_path, segmentation_original)
    print(f"Saved segmented image to: {save_path}")

    # Show results
    visualize_results(args.image, segmentation)


if __name__ == "__main__":
    main()


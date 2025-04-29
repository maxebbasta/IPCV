#!/usr/bin/env python3
"""
Synthetic Dataset Generator for YOLOv5

This script creates a synthetic dataset by compositing product model cutouts onto real shelf backgrounds.
For each generated image, it randomly selects a background, places a number of product instances with random
transformations, and writes corresponding YOLO-format labels.

Usage:
    python generate_synthetic_dataset.py \
        --backgrounds_dir path/to/backgrounds \
        --models_dir path/to/models_png \
        --output_images_dir output/images \
        --output_labels_dir output/labels \
        --num_images 1000 \
        --min_objs 5 --max_objs 15

Requirements:
    pip install opencv-python numpy
"""
import os
import cv2
import numpy as np
import argparse
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for YOLOv5")
    parser.add_argument('--backgrounds_dir', type=str, required=True,
                        help='Directory with background images (shelf photos)')
    parser.add_argument('--models_dir', type=str, required=True,
                        help='Directory with model PNGs (with alpha) named by class id, e.g. 0.png, 1.png')
    parser.add_argument('--output_images_dir', type=str, required=True,
                        help='Directory to save generated images')
    parser.add_argument('--output_labels_dir', type=str, required=True,
                        help='Directory to save YOLO-format labels')
    parser.add_argument('--num_images', type=int, default=500,
                        help='Total number of synthetic images to generate')
    parser.add_argument('--min_objs', type=int, default=3,
                        help='Minimum number of objects per image')
    parser.add_argument('--max_objs', type=int, default=10,
                        help='Maximum number of objects per image')
    return parser.parse_args()


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_models(models_dir):
    models = {}
    for fname in os.listdir(models_dir):
        if not (fname.lower().endswith('.png')):
            continue
        cls = os.path.splitext(fname)[0]
        path = os.path.join(models_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # BGRA
        if img is None or img.shape[2] != 4:
            continue
        models[int(cls)] = img
    return models


def overlay_image(bg, fg, x, y):
    """Overlay fg (with alpha channel) onto bg at position (x, y)"""
    h, w = fg.shape[:2]
    # Ensure ROI inside background
    bh, bw = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + w), min(bh, y + h)
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

    alpha = fg[fy1:fy2, fx1:fx2, 3:] / 255.0
    fg_rgb = fg[fy1:fy2, fx1:fx2, :3]
    roi = bg[y1:y2, x1:x2]
    blended = (1 - alpha) * roi + alpha * fg_rgb
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return (x1, y1, x2, y2)


def main():
    args = parse_args()
    ensure_dir(args.output_images_dir)
    ensure_dir(args.output_labels_dir)

    # Load backgrounds and models
    bg_paths = [os.path.join(args.backgrounds_dir, f) for f in os.listdir(args.backgrounds_dir)
                if f.lower().endswith(('.jpg','.png'))]
    models = load_models(args.models_dir)
    assert models, "No model PNGs found in models_dir"

    for idx in range(args.num_images):
        # Select random background
        bg_path = random.choice(bg_paths)
        bg = cv2.imread(bg_path)
        if bg is None:
            continue
        h_bg, w_bg = bg.shape[:2]
        labels = []

        num_objs = random.randint(args.min_objs, args.max_objs)
        for _ in range(num_objs):
            # Choose random model
            cls, fg = random.choice(list(models.items()))
            # Random scale & rotation
            scale = random.uniform(0.5, 1.2)
            angle = random.uniform(-15, 15)
            fh, fw = fg.shape[:2]
            new_w, new_h = int(fw * scale), int(fh * scale)
            fg_resized = cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Rotation
            M = cv2.getRotationMatrix2D((new_w/2, new_h/2), angle, 1.0)
            fg_rot = cv2.warpAffine(fg_resized, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

            # Random position
            x = random.randint(0, max(0, w_bg - new_w))
            y = random.randint(0, max(0, h_bg - new_h))

            # Overlay and get actual bbox used
            x1,y1,x2,y2 = overlay_image(bg, fg_rot, x, y)
            # Compute YOLO format: normalized center x,y,w,h
            cx = ((x1 + x2) / 2) / w_bg
            cy = ((y1 + y2) / 2) / h_bg
            bw = (x2 - x1) / w_bg
            bh = (y2 - y1) / h_bg
            labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Save image and label
        img_name = f"synthetic_{idx:05d}.jpg"
        label_name = f"synthetic_{idx:05d}.txt"
        cv2.imwrite(os.path.join(args.output_images_dir, img_name), bg)
        with open(os.path.join(args.output_labels_dir, label_name), 'w') as f:
            f.write("\n".join(labels))

        if idx % 100 == 0:
            print(f"Generated {idx} / {args.num_images}")

    print("Dataset generation complete.")

if __name__ == '__main__':
    main()

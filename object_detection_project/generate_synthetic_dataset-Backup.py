#!/usr/bin/env python3
"""
Synthetic Dataset Generator for YOLOv5

This script creates a synthetic dataset by compositing product images (JPG/PNG) onto real shelf backgrounds.
It ensures each class appears at least a fixed number of times and prevents excessive overlap between instances.
Rotations are performed on a padded canvas to avoid clipping (no more hexagonal/rectangular artifacts).
Directories and parameters are hardcoded below.
"""
import os
import cv2
import numpy as np
import random

# ------------------------
# Configuration
# ------------------------
BACKGROUNDS_DIR     = "dataset/backgrounds"   # sfondi (1.jpg–25.jpg)
MODELS_DIR          = "models"                # ritagli modello (0.jpg,…,23.jpg)
OUTPUT_IMAGES_DIR   = "dataset/images/train"  # dove salvare immagini sintetiche
OUTPUT_LABELS_DIR   = "dataset/labels/train"  # dove salvare etichette YOLO
NUM_IMAGES          = 2500                      # numero minimo di immagini da generare
MIN_OBJS, MAX_OBJS  = 5, 12                   # min/max oggetti per immagine
MIN_OCC_PER_CLASS   = 100                      # occorrenze minime per ogni classe
MAX_IOU             = 0.1                      # soglia IoU massima per evitare sovrapposizioni
MIN_FACTOR, MAX_FACTOR = 4.0, 5.0              # model height = bg_height / factor
# ------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    ix1 = max(x1, xx1); iy1 = max(y1, yy1)
    ix2 = min(x2, xx2); iy2 = min(y2, yy2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_models(models_dir):
    models = {}
    for fname in os.listdir(models_dir):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        base = os.path.splitext(fname)[0].split('_')[0]
        try:
            cls = int(base)
        except ValueError:
            continue
        img = cv2.imread(os.path.join(models_dir, fname), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        # ensure RGBA
        if img.ndim == 3 and img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.full_like(b, 255)
            img = cv2.merge((b, g, r, alpha))
        elif not (img.ndim == 3 and img.shape[2] == 4):
            continue
        models[cls] = img
    if not models:
        raise RuntimeError(f"No model images found in {models_dir}")
    return models


def overlay_image(bg, fg, x, y):
    h_fg, w_fg = fg.shape[:2]
    h_bg, w_bg = bg.shape[:2]
    x1 = max(0, x); y1 = max(0, y)
    x2 = min(w_bg, x + w_fg); y2 = min(h_bg, y + h_fg)
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    alpha = fg[fy1:fy2, fx1:fx2, 3:] / 255.0
    fg_rgb = fg[fy1:fy2, fx1:fx2, :3]
    roi = bg[y1:y2, x1:x2]
    bg[y1:y2, x1:x2] = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)
    return x1, y1, x2, y2


def rotate_full(fg, angle):
    # rotate fg on expanded canvas to avoid clipping
    h, w = fg.shape[:2]
    theta = np.deg2rad(angle)
    cos_t, sin_t = abs(np.cos(theta)), abs(np.sin(theta))
    new_w = int(w * cos_t + h * sin_t)
    new_h = int(w * sin_t + h * cos_t)
    # rotation matrix around center
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    # adjust translation to center in new canvas
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    # warp into new canvas
    return cv2.warpAffine(fg, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))


def main():
    ensure_dir(OUTPUT_IMAGES_DIR)
    ensure_dir(OUTPUT_LABELS_DIR)

    bg_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not bg_files:
        raise RuntimeError(f"No background images in {BACKGROUNDS_DIR}")
    models = load_models(MODELS_DIR)

    counts = {cls: 0 for cls in models.keys()}
    idx = 0
    while idx < NUM_IMAGES or any(counts[c] < MIN_OCC_PER_CLASS for c in counts):
        bg = cv2.imread(os.path.join(BACKGROUNDS_DIR, random.choice(bg_files)))
        if bg is None:
            continue
        h_bg, w_bg = bg.shape[:2]
        labels, bboxes = [], []
        num_objs = random.randint(MIN_OBJS, MAX_OBJS)
        for _ in range(num_objs):
            cls, fg_orig = random.choice(list(models.items()))
            fh, fw = fg_orig.shape[:2]
            # scale height
            factor = random.uniform(MIN_FACTOR, MAX_FACTOR)
            target_h = int(h_bg / factor)
            scale = target_h / float(fh)
            fg = cv2.resize(fg_orig, (int(fw * scale), target_h), interpolation=cv2.INTER_AREA)
            # rotate on full canvas
            fg = rotate_full(fg, random.uniform(-15, 15))
            h_fg, w_fg = fg.shape[:2]
            # placement
            for _ in range(10):
                x = random.randint(0, max(0, w_bg - w_fg))
                y = random.randint(0, max(0, h_bg - h_fg))
                box = (x, y, x + w_fg, y + h_fg)
                if all(compute_iou(box, bb) < MAX_IOU for bb in bboxes):
                    x1, y1, x2, y2 = overlay_image(bg, fg, x, y)
                    bboxes.append((x1, y1, x2, y2))
                    counts[cls] += 1
                    cx = ((x1 + x2) / 2) / w_bg
                    cy = ((y1 + y2) / 2) / h_bg
                    bw = (x2 - x1) / w_bg
                    bh = (y2 - y1) / h_bg
                    labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    break
        # save outputs
        name = f"synthetic_{idx:05d}"
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{name}.jpg"), bg)
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{name}.txt"), 'w') as f:
            f.write("\n".join(labels))
        idx += 1
        if idx % 50 == 0:
            print(f"Generated {idx}/{NUM_IMAGES}, counts min={min(counts.values())}")
    print("Dataset generation complete.")
    print("Final counts per class:", counts)

if __name__ == '__main__':
    main()

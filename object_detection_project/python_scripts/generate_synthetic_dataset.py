#!/usr/bin/env python3
"""
Synthetic Dataset Generator for YOLOv5

This script creates a synthetic dataset by compositing product images (JPG/PNG) onto real shelf backgrounds.
It ensures each class appears at least a fixed number of times and prevents excessive overlap between instances.
Augmentations include color jitter, blur/noise, perspective warp, edge-cut occlusion and random erasing to improve robustness.
"""
import os
import cv2
import numpy as np
import random

# ------------------------
# Configuration
# ------------------------
BACKGROUNDS_DIR     = "dataset/backgrounds"   # background images (e.g. 1.jpg–25.jpg)
MODELS_DIR          = "models"                # product cutouts (0.jpg,…,23.jpg) with alpha channel
OUTPUT_IMAGES_DIR   = "dataset/images/train"  # folder where synthetic images will be saved
OUTPUT_LABELS_DIR   = "dataset/labels/train"  # folder where YOLO label files will be saved
# NUM_IMAGES          = 30000                   # minimum number of synthetic images to generate
# MIN_OBJS, MAX_OBJS  = 40, 60                  # min/max number of objects per image
# MIN_OCC_PER_CLASS   = 35000                   # minimum total occurrences per class in the whole dataset
NUM_IMAGES          = 5
MIN_OBJS, MAX_OBJS  = 40, 60
MIN_OCC_PER_CLASS   = 1
MAX_IOU             = 0.15                    # maximum allowed IoU between objects (to avoid excessive overlap)
MIN_FACTOR, MAX_FACTOR = 4.5, 6.0             # object height = bg_height / factor (controls relative object size)
# ------------------------


def ensure_dir(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Boxes are given as (x1, y1, x2, y2) in absolute pixel coordinates.
    """
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    # Intersection coordinates
    ix1, iy1 = max(x1, xx1), max(y1, yy1)
    ix2, iy2 = min(x2, xx2), min(y2, yy2)

    # Intersection width/height
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    # Individual box areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def load_models(models_dir):
    """
    Load product models from `models_dir`.

    Expected file format:
    - Filenames begin with an integer class ID, e.g. "0.png", "3_variant.png", etc.
    - Images can be JPG or PNG.
    - If an image has only 3 channels (BGR), an opaque alpha channel is added.
    - Only images with 4 channels (BGRA) are kept.

    Returns:
        dict: class_id -> RGBA image (numpy array).
    """
    models = {}
    for fname in os.listdir(models_dir):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        # Extract class ID from filename (everything before first underscore)
        base = os.path.splitext(fname)[0].split('_')[0]
        try:
            cls = int(base)
        except ValueError:
            # Skip files that do not start with a valid integer class ID
            continue

        img_path = os.path.join(models_dir, fname)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue

        # Ensure there is an alpha channel (BGRA)
        if img.ndim == 3 and img.shape[2] == 3:
            # Add a full opaque alpha channel
            b, g, r = cv2.split(img)
            alpha = np.full_like(b, 255)
            img = cv2.merge((b, g, r, alpha))
        elif not (img.ndim == 3 and img.shape[2] == 4):
            # Skip images that are not 3- or 4-channel
            continue

        models[cls] = img

    if not models:
        raise RuntimeError(f"No model images found in {models_dir}")
    return models


def overlay_image(bg, fg, x, y):
    """
    Alpha-blend RGBA foreground `fg` onto BGR background `bg` at position (x, y).

    Args:
        bg (np.ndarray): background image (H x W x 3)
        fg (np.ndarray): foreground image (h x w x 4) with alpha channel
        x (int): top-left x-coordinate on background
        y (int): top-left y-coordinate on background

    Returns:
        tuple: (x1, y1, x2, y2) final bounding box coordinates of the placed foreground.
    """
    h_fg, w_fg = fg.shape[:2]
    h_bg, w_bg = bg.shape[:2]

    # Clip to background boundaries
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + w_fg), min(h_bg, y + h_fg)

    # Corresponding region on the foreground
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

    # Extract alpha and RGB from foreground
    alpha = fg[fy1:fy2, fx1:fx2, 3:] / 255.0
    fg_rgb = fg[fy1:fy2, fx1:fx2, :3]

    # Region of interest on background
    roi = bg[y1:y2, x1:x2]

    # Alpha blending
    bg[y1:y2, x1:x2] = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)

    return x1, y1, x2, y2


def rotate_full(fg, angle):
    """
    Rotate an RGBA image `fg` by `angle` degrees, expanding the canvas so that
    the entire rotated object fits without cropping.
    """
    h, w = fg.shape[:2]
    theta = np.deg2rad(angle)
    cos_t, sin_t = abs(np.cos(theta)), abs(np.sin(theta))

    # New bounding dimensions after rotation
    new_w = int(w * cos_t + h * sin_t)
    new_h = int(w * sin_t + h * cos_t)

    # Rotation matrix around the original center
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # Adjust translation to re-center on the new canvas
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        fg,
        M,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def random_perspective(img, max_warp=0.2):
    """
    Apply a random perspective transformation to `img`.

    The four corners of the image are randomly displaced by up to `max_warp`
    times the width/height, then a homography is computed.

    Args:
        img (np.ndarray): input RGBA image
        max_warp (float): maximum relative displacement of corners

    Returns:
        np.ndarray: perspective-warped RGBA image.
    """
    h, w = img.shape[:2]

    def pts(delta):
        return [
            (random.uniform(-delta, delta) * w, random.uniform(-delta, delta) * h),
            (w + random.uniform(-delta, delta) * w, random.uniform(-delta, delta) * h),
            (w + random.uniform(-delta, delta) * w, h + random.uniform(-delta, delta) * h),
            (random.uniform(-delta, delta) * w, h + random.uniform(-delta, delta) * h),
        ]

    src = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
    dst = np.float32(pts(max_warp))
    M = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(
        img,
        M,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def main():
    """Main entry point: generate synthetic images and YOLO labels."""
    # Ensure output directories exist
    ensure_dir(OUTPUT_IMAGES_DIR)
    ensure_dir(OUTPUT_LABELS_DIR)

    # Collect all background files
    bg_files = [
        f for f in os.listdir(BACKGROUNDS_DIR)
        if f.lower().endswith(('.jpg', '.png'))
    ]
    if not bg_files:
        raise RuntimeError(f"No background images in {BACKGROUNDS_DIR}")

    # Load product models (class_id -> RGBA image)
    models = load_models(MODELS_DIR)

    # Track how many times each class has been placed
    counts = {cls: 0 for cls in models.keys()}

    idx = 0
    # Continue until:
    #   - at least NUM_IMAGES images generated, AND
    #   - each class has at least MIN_OCC_PER_CLASS occurrences
    while idx < NUM_IMAGES or any(counts[c] < MIN_OCC_PER_CLASS for c in counts):
        # Randomly select a background image
        bg_path = os.path.join(BACKGROUNDS_DIR, random.choice(bg_files))
        bg = cv2.imread(bg_path)
        if bg is None:
            continue

        h_bg, w_bg = bg.shape[:2]
        labels, bboxes = [], []

        # Random number of objects for this image
        num_objs = random.randint(MIN_OBJS, MAX_OBJS)

        for _ in range(num_objs):
            # Randomly sample a model (class_id and RGBA image)
            cls, fg_orig = random.choice(list(models.items()))
            fh, fw = fg_orig.shape[:2]

            # --- Scaling: control size relative to background height ---
            factor = random.uniform(MIN_FACTOR, MAX_FACTOR)
            target_h = int(h_bg / factor)
            scale = target_h / float(fh)
            fg = cv2.resize(
                fg_orig,
                (int(fw * scale), target_h),
                interpolation=cv2.INTER_AREA
            )

            # --- Split RGB and alpha channels ---
            bgr = fg[..., :3]
            alpha_ch = fg[..., 3:]

            # --- Color jitter: brightness/contrast + saturation ---
            a = random.uniform(0.8, 1.2)       # contrast
            b = random.uniform(-30, 30)        # brightness
            bgr = cv2.convertScaleAbs(bgr, alpha=a, beta=b)

            # HSV-based saturation change
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.7, 1.3)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Merge back RGB and alpha
            fg = np.dstack((bgr, alpha_ch))

            # --- Rotation ---
            fg = rotate_full(fg, random.uniform(-20, 20))

            # --- Edge-cut occlusion (simulate partial occlusions) ---
            if random.random() < 0.5:
                h_fg, w_fg = fg.shape[:2]
                side = random.choice([0, 1, 2, 3])  # 0=top,1=right,2=bottom,3=left
                depth = int(
                    (h_fg if side in [0, 2] else w_fg) * random.uniform(0.05, 0.70)
                )
                if side == 0:       # cut from top
                    fg[:depth, :, 3] = 0
                elif side == 2:     # cut from bottom
                    fg[-depth:, :, 3] = 0
                elif side == 1:     # cut from right
                    fg[:, -depth:, 3] = 0
                else:               # cut from left
                    fg[:, :depth, 3] = 0

            # --- Blur (simulated defocus/motion blur) ---
            if random.random() < 0.3:
                k = random.choice([3, 5])
                bgr = cv2.GaussianBlur(fg[..., :3], (k, k), 0)
                fg = np.dstack((bgr, fg[..., 3:]))

            # --- Additive Gaussian noise ---
            if random.random() < 0.2:
                noise = np.random.normal(0, 10, fg[..., :3].shape).astype(np.int16)
                bgr = np.clip(
                    fg[..., :3].astype(np.int16) + noise,
                    0,
                    255
                ).astype(np.uint8)
                fg = np.dstack((bgr, fg[..., 3:]))

            # --- Perspective warp (viewpoint changes) ---
            if random.random() < 0.3:
                fg = random_perspective(fg)

            h_fg, w_fg = fg.shape[:2]

            # Try up to 10 random positions to find one with low IoU
            for _ in range(10):
                x = random.randint(0, max(0, w_bg - w_fg))
                y = random.randint(0, max(0, h_bg - h_fg))

                # Candidate bounding box in absolute coordinates
                box = (x, y, x + w_fg, y + y + h_fg)

                # Check IoU against already placed objects
                if all(compute_iou(box, bb) < MAX_IOU for bb in bboxes):
                    # Place object and update counts/labels
                    x1, y1, x2, y2 = overlay_image(bg, fg, x, y)
                    bboxes.append((x1, y1, x2, y2))
                    counts[cls] += 1

                    # Convert to YOLO normalized format
                    cx = ((x1 + x2) / 2) / w_bg
                    cy = ((y1 + y2) / 2) / h_bg
                    bw = (x2 - x1) / w_bg
                    bh = (y2 - y1) / h_bg

                    labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    break

        # Image base name (shared between image and label)
        name = f"synthetic_{idx:05d}"

        # Force height = 640 px for all images (keep aspect ratio)
        h_bg, w_bg = bg.shape[:2]
        new_h = 640
        new_w = int(w_bg * new_h / h_bg)
        bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Save image and corresponding YOLO label file
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{name}.jpg"), bg)
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{name}.txt"), 'w') as f:
            f.write("\n".join(labels))

        idx += 1

        # Periodic progress report
        if idx % 50 == 0:
            print(f"Generated {idx}/{NUM_IMAGES}, min count={min(counts.values())}")

    print("Dataset generation complete.")
    print("Final counts per class:", counts)


if __name__ == '__main__':
    main()
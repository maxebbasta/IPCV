import os
import cv2
import numpy as np
import random

BACKGROUNDS_DIR     = "dataset/backgrounds"
MODELS_DIR          = "models"
OUTPUT_IMAGES_DIR   = "dataset/images/train"  
OUTPUT_LABELS_DIR   = "dataset/labels/train"  

NUM_IMAGES          = 5
MIN_OBJS, MAX_OBJS  = 40, 60
MIN_OCC_PER_CLASS   = 1
MAX_IOU             = 0.15
MIN_FACTOR, MAX_FACTOR = 4.5, 6.0


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[SETUP] Created directory: {path}")


def compute_iou(box1, box2):

    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2

    ix1, iy1 = max(x1, xx1), max(y1, yy1)
    ix2, iy2 = min(x2, xx2), min(y2, yy2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

def load_models(models_dir):
    models = {}
    print(f"[INFO] Loading product templates from {models_dir}...")
    
    for fname in os.listdir(models_dir):
        if not fname.lower().endswith('.jpg'):
            continue

        try:
            cls = int(os.path.splitext(fname)[0])
        except ValueError:
            continue

        img_path = os.path.join(models_dir, fname)
        
        img = cv2.imread(img_path)
        
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

        models[cls] = img

    if not models:
        raise RuntimeError(f"No valid .jpg images found in {models_dir}")
    
    print(f"[INFO] Loaded {len(models)} classes.")
    return models

def overlay_image(bg_img, fg_img, x, y):

    h_fg, w_fg = fg_img.shape[:2]
    h_bg, w_bg = bg_img.shape[:2]

    # Clipping coordinates to stay within background
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + w_fg), min(h_bg, y + h_fg)

    # Offset for foreground cropping
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

    # Alpha mask and normalize to 0-1 range
    alpha = fg_img[fy1:fy2, fx1:fx2, 3:] / 255.0
    fg_rgb = fg_img[fy1:fy2, fx1:fx2, :3]

    roi = bg_img[y1:y2, x1:x2]

    # Blend
    bg_img[y1:y2, x1:x2] = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)

    return x1, y1, x2, y2

def rotate_full(image, angle):

    h, w = image.shape[:2]
    theta = np.deg2rad(angle)
    cos_t, sin_t = abs(np.cos(theta)), abs(np.sin(theta))

    # New canvas dimensions
    new_w = int(w * cos_t + h * sin_t)
    new_h = int(w * sin_t + h * cos_t)

    # Rotation matrix centered in the image
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    
    # Add translation to the matrix to center the result
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(
        image, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) # Transparent border
    )

def random_perspective(img, max_warp=0.2):

    h, w = img.shape[:2]

    # Random displacement for corners
    def get_jitter(limit):
        return random.uniform(-limit, limit)

    # Corners of original image
    src_pts = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
    
    # Perturbed corners destination
    dst_pts = np.float32([
        (get_jitter(max_warp) * w, get_jitter(max_warp) * h),           # Top-left
        (w + get_jitter(max_warp) * w, get_jitter(max_warp) * h),       # Top-right
        (w + get_jitter(max_warp) * w, h + get_jitter(max_warp) * h),   # Bottom-right
        (get_jitter(max_warp) * w, h + get_jitter(max_warp) * h),       # Bottom-left
    ])

    # Homography Matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    return cv2.warpPerspective(
        img, M, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0)
    )

def main():

    ensure_dir(OUTPUT_IMAGES_DIR)
    ensure_dir(OUTPUT_LABELS_DIR)

    bg_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.jpg', '.png'))]
    if not bg_files:
        raise RuntimeError(f"No background images found in {BACKGROUNDS_DIR}")

    models = load_models(MODELS_DIR)
    counts = {cls: 0 for cls in models.keys()} # Time a model appears

    idx = 0
    print(f"[START] Generating {NUM_IMAGES} synthetic scenes...")

    while idx < NUM_IMAGES or any(counts[c] < MIN_OCC_PER_CLASS for c in counts):
        
        # Background Selection
        bg_path = os.path.join(BACKGROUNDS_DIR, random.choice(bg_files))
        shelf_bg = cv2.imread(bg_path)
        if shelf_bg is None: continue

        h_bg, w_bg = shelf_bg.shape[:2]
        current_labels = []
        current_bboxes = []

        num_objs = random.randint(MIN_OBJS, MAX_OBJS)

        # Product Placement
        for _ in range(num_objs):
            cls_id, template_orig = random.choice(list(models.items()))
            fh, fw = template_orig.shape[:2]

            # Object size
            factor = random.uniform(MIN_FACTOR, MAX_FACTOR)
            target_h = int(h_bg / factor)
            scale = target_h / float(fh)
            
            product_img = cv2.resize(template_orig, (int(fw * scale), target_h), interpolation=cv2.INTER_AREA)

            # Split channels to operate on RGB only
            rgb_layer = product_img[..., :3]
            alpha_layer = product_img[..., 3:]

            # Brightness/Contrast adjustment
            contrast = random.uniform(0.8, 1.2)
            brightness = random.uniform(-30, 30)
            rgb_layer = cv2.convertScaleAbs(rgb_layer, alpha=contrast, beta=brightness)

            # HSV Saturation adjustament
            hsv = cv2.cvtColor(rgb_layer, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= random.uniform(0.7, 1.3)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            rgb_layer = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Recombine
            product_img = np.dstack((rgb_layer, alpha_layer))

            # Rotation
            product_img = rotate_full(product_img, random.uniform(-20, 20))

            # Occlusion
            if random.random() < 0.5:
                h_p, w_p = product_img.shape[:2]
                side = random.choice([0, 1, 2, 3]) # Top, Right, Bottom, Left
                depth = int((h_p if side in [0, 2] else w_p) * random.uniform(0.05, 0.70))
                
                if side == 0: product_img[:depth, :, 3] = 0
                elif side == 2: product_img[-depth:, :, 3] = 0
                elif side == 1: product_img[:, -depth:, 3] = 0
                else: product_img[:, :depth, 3] = 0

            # Low quality
            if random.random() < 0.3:
                k = random.choice([3, 5])
                blurred = cv2.GaussianBlur(product_img[..., :3], (k, k), 0)
                product_img = np.dstack((blurred, product_img[..., 3:]))

            if random.random() < 0.2:
                noise = np.random.normal(0, 10, product_img[..., :3].shape).astype(np.int16)
                noisy_rgb = np.clip(product_img[..., :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)
                product_img = np.dstack((noisy_rgb, product_img[..., 3:]))

            # Perspective Transform
            if random.random() < 0.3:
                product_img = random_perspective(product_img)

            # Placement Check
            h_final, w_final = product_img.shape[:2]

            # Free spot - max 10 attempts
            for attempt in range(10):
                x = random.randint(0, max(0, w_bg - w_final))
                y = random.randint(0, max(0, h_bg - h_final))

                candidate_box = (x, y, x + w_final, y + y + h_final)

                # Check overlap against previously placed objects
                collision = False
                for existing_box in current_bboxes:
                    if compute_iou(candidate_box, existing_box) > MAX_IOU:
                        collision = True
                        break
                
                if not collision:
                    # Place object
                    x1, y1, x2, y2 = overlay_image(shelf_bg, product_img, x, y)
                    current_bboxes.append((x1, y1, x2, y2))
                    counts[cls_id] += 1

                    # YOLO Label
                    cx = ((x1 + x2) / 2) / w_bg
                    cy = ((y1 + y2) / 2) / h_bg
                    bw = (x2 - x1) / w_bg
                    bh = (y2 - y1) / h_bg

                    current_labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    break 

        # Save
        img_filename = f"synthetic_{idx:05d}"
        
        # Resizing size
        final_h = 640
        final_w = int(w_bg * final_h / h_bg)
        shelf_bg_resized = cv2.resize(shelf_bg, (final_w, final_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{img_filename}.jpg"), shelf_bg_resized)
        
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{img_filename}.txt"), 'w') as f_label:
            f_label.write("\n".join(current_labels))

        idx += 1
        if idx % 50 == 0:
            print(f"[STATUS] Generated {idx}/{NUM_IMAGES} images. Class min count: {min(counts.values())}")

    print(f"[COMPLETE] Final dataset statistics: {counts}")

if __name__ == '__main__':
    main()
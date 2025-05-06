#!/usr/bin/env python3
"""
Synthetic Dataset Generator per YOLOv5 con partial/full-height placement.

– Oggetti possono apparire per il 10 % nel frame (anche spigoli tagliati)
– Modalità a random: full-height / partial-edge / normale
– Calcolo corretto delle bbox clippate

Note:
- Non è necessario un folder di backgrounds reali: usiamo sfondi procedurali a scaffale.
- Assicurati di installare Albumentations >=1.0.0:
  `pip install albumentations>=1.0.0`
"""
import os
import cv2
import numpy as np
import random
import albumentations as A

# ------------------------
# Configurazione
# ------------------------
MODELS_DIR             = "models"                 # cartella con ritagli PNG RGBA
OUTPUT_IMAGES_DIR      = "dataset/images/train"   # dove salvare le immagini
OUTPUT_LABELS_DIR      = "dataset/labels/train"   # dove salvare le label
NUM_IMAGES             = 5000
MIN_OBJS, MAX_OBJS     = 15, 20                    # oggetti per immagine
MIN_OCC_PER_CLASS      = 2500                      # occorrenze minime per classe
MAX_IOU                = 0.1                       # IoU max tra box
MIN_FACTOR, MAX_FACTOR = 5, 7                      # altezza oggetto = h_bg/factor
IMAGE_SIZE             = 640                       # dimensione quadrata sfondo

# ------------------------
# Helpers
# ------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# calcola IoU fra due bbox (x1,y1,x2,y2)
def compute_iou(a, b):
    x1, y1, x2, y2 = a
    xx1, yy1, xx2, yy2 = b
    iw = max(0, min(x2, xx2) - max(x1, xx1))
    ih = max(0, min(y2, yy2) - max(y1, yy1))
    inter = iw * ih
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (xx2 - xx1) * (yy2 - yy1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0

# carica tutti i ritagli (RGBA o RGB)
def load_models(models_dir):
    models = {}
    for fn in os.listdir(models_dir):
        if not fn.lower().endswith(('.png', '.jpg')):
            continue
        try:
            cls = int(os.path.splitext(fn)[0].split('_')[0])
        except:
            continue
        img = cv2.imread(os.path.join(models_dir, fn), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        models[cls] = img
    if not models:
        raise RuntimeError(f"Nessun modello caricato in {models_dir}")
    return models

# overlay FG su BG con alpha-check
# restituisce bbox (x1,y1,x2,y2) o None se nessun overlap

def overlay_image(bg, fg, x, y):
    h_fg, w_fg = fg.shape[:2]
    h_bg, w_bg = bg.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_bg, x + w_fg), min(h_bg, y + h_fg)
    if x1 >= x2 or y1 >= y2:
        return None
    fx1, fy1 = x1 - x, y1 - y
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)
    crop_fg = fg[fy1:fy2, fx1:fx2]
    # gestisci alpha o RGB
    if crop_fg.ndim == 2:
        crop_fg = cv2.cvtColor(crop_fg, cv2.COLOR_GRAY2BGR)
        alpha = np.ones((crop_fg.shape[0], crop_fg.shape[1], 1), dtype=np.float32)
        fg_rgb = crop_fg
    elif crop_fg.shape[2] == 4:
        fg_rgb = crop_fg[..., :3]
        alpha = crop_fg[..., 3:] / 255.0
    else:
        fg_rgb = crop_fg
        alpha = np.ones((crop_fg.shape[0], crop_fg.shape[1], 1), dtype=np.float32)
    roi = bg[y1:y2, x1:x2]
    # ridimensiona se occorre
    h0, w0 = fg_rgb.shape[:2]
    roi = roi[:h0, :w0]
    alpha = alpha[:h0, :w0]
    fg_rgb = fg_rgb[:h0, :w0]
    blended = (alpha * fg_rgb + (1 - alpha) * roi).astype(np.uint8)
    bg[y1:y1+h0, x1:x1+w0] = blended
    return x1, y1, x1+w0, y1+h0

# ombra procedurale tramite poligono sfocato
def add_random_shadow(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array([[
        (random.randint(0, w), random.randint(0, h)),
        (random.randint(0, w), random.randint(0, h)),
        (random.randint(0, w), random.randint(0, h)),
        (random.randint(0, w), random.randint(0, h))
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    mask = cv2.GaussianBlur(mask, (101, 101), 0) / 255.0
    for c in range(3):
        img[:, :, c] = (img[:, :, c] * (1 - 0.5 * mask)).astype(np.uint8)
    return img

# occlusioni fittizie (cutout)
def add_random_cutout(img):
    h, w = img.shape[:2]
    for _ in range(random.randint(1, 5)):
        x1 = random.randint(0, w - 1)
        y1 = random.randint(0, h - 1)
        x2 = min(w, x1 + random.randint(20, w // 4))
        y2 = min(h, y1 + random.randint(20, h // 4))
        color = img[y1:y2, x1:x2].mean(axis=(0, 1)).astype(np.uint8)
        cv2.rectangle(img, (x1, y1), (x2, y2), tuple(int(c) for c in color), -1)
    return img

# pipeline Albumentations globale (photometriche)
augment = A.Compose([
    A.OneOf([
        A.MotionBlur(blur_limit=7, p=0.3),
        A.MedianBlur(blur_limit=5, p=0.2),
        A.GaussianBlur(blur_limit=5, p=0.3),
    ], p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
    A.GaussNoise(p=0.3),
    A.ImageCompression(p=0.4),
], p=1.0)

# trasformazioni geometriche su FG
def rotate_full(fg, angle):
    h, w = fg.shape[:2]
    theta = np.deg2rad(angle)
    cos, sin = abs(np.cos(theta)), abs(np.sin(theta))
    nw, nh = int(w * cos + h * sin), int(w * sin + h * cos)
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    M[0,2] += (nw - w)/2
    M[1,2] += (nh - h)/2
    return cv2.warpAffine(fg, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

def random_perspective(img, max_warp=0.05):
    h, w = img.shape[:2]
    src = np.float32([(0,0), (w,0), (w,h), (0,h)])
    dst = np.float32([
        (random.uniform(-max_warp, max_warp)*w, random.uniform(-max_warp, max_warp)*h),
        (w + random.uniform(-max_warp, max_warp)*w, random.uniform(-max_warp, max_warp)*h),
        (w + random.uniform(-max_warp, max_warp)*w, h + random.uniform(-max_warp, max_warp)*h),
        (random.uniform(-max_warp, max_warp)*w, h + random.uniform(-max_warp, max_warp)*h)
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

# ------------------------
# Procedural Background Generator
# ------------------------
def generate_shelf_background(width, height, min_shelves=3, max_shelves=6):
    """
    Genera sfondo simulando scaffali con linee e rumore.
    """
    base_choices = [np.array([200,200,200]), np.array([180,180,180]), np.array([160,160,160])]
    base = base_choices[random.randint(0,len(base_choices)-1)].astype(np.uint8)
    bg = np.ones((height, width, 3), dtype=np.uint8) * base
    num_shelves = random.randint(min_shelves, max_shelves)
    shelf_h = height // num_shelves
    for i in range(1, num_shelves):
        y = int(i*shelf_h + random.uniform(-shelf_h*0.1, shelf_h*0.1))
        thickness = random.randint(4,8)
        shade = (base * random.uniform(0.6,0.8)).astype(np.uint8)
        cv2.rectangle(bg, (0,y), (width, y+thickness), tuple(int(c) for c in shade), -1)
    bg = cv2.GaussianBlur(bg, (5,5), 0)
    noise = np.zeros((height, width), dtype=np.int16)
    cv2.randn(noise,0,10)
    for c in range(3):
        ch = bg[:,:,c].astype(np.int16) + noise
        bg[:,:,c] = np.clip(ch,0,255).astype(np.uint8)
    return bg

# ------------------------
# Main
# ------------------------
def main():
    ensure_dir(OUTPUT_IMAGES_DIR)
    ensure_dir(OUTPUT_LABELS_DIR)
    models = load_models(MODELS_DIR)
    counts = {c:0 for c in models}
    idx = 0
    while idx < NUM_IMAGES or any(count<MIN_OCC_PER_CLASS for count in counts.values()):
        # genera sfondo procedurale
        bg = generate_shelf_background(IMAGE_SIZE, IMAGE_SIZE)
        h_bg, w_bg = IMAGE_SIZE, IMAGE_SIZE
        labels, bboxes = [], []
        n_obj = random.randint(MIN_OBJS, MAX_OBJS)
        for _ in range(n_obj):
            cls, fg0 = random.choice(list(models.items()))
            fh, fw = fg0.shape[:2]
            mode = random.random()
            factor = random.uniform(1.75,3.5) if mode<0.05 else random.uniform(MIN_FACTOR,MAX_FACTOR)
            th = int(h_bg/factor)
            scale = th/fh
            fg = cv2.resize(fg0,(int(fw*scale),th),interpolation=cv2.INTER_AREA)
            # HSV jitter su FG
            if fg.ndim==2:
                fg = cv2.cvtColor(fg,cv2.COLOR_GRAY2BGR)
                a_ch=None
            elif fg.shape[2]==4:
                bgr, a_ch = fg[..., :3], fg[...,3:]/255.0
            else:
                bgr, a_ch = fg[..., :3], None
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[...,1] = np.clip(hsv[...,1]*random.uniform(0.7,1.3),0,255)
            bgr = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            fg = np.dstack((bgr,a_ch)) if a_ch is not None else bgr
            # geometric
            fg = rotate_full(fg, random.uniform(-15,15))
            if random.random()<0.3:
                fg = random_perspective(fg)
            # overlay
            result = overlay_image(bg, fg, random.randint(int(-0.9*fg.shape[1]),int(w_bg-0.1*fg.shape[1])),
                                           random.randint(int(-0.9*fg.shape[0]),int(h_bg-0.1*fg.shape[0])))
            if not result:
                continue
            x1,y1,x2,y2 = result
            if x2-x1<2 or y2-y1<2:
                continue
            bboxes.append((x1,y1,x2,y2))
            counts[cls]+=1
            cx, cy = ((x1+x2)/2)/w_bg, ((y1+y2)/2)/h_bg
            bw, bh = (x2-x1)/w_bg, (y2-y1)/h_bg
            labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        # augmentazioni globali
        bg = add_random_shadow(bg)
        bg = add_random_cutout(bg)
        bg = augment(image=bg)['image']
        # salva
        name = f"synthetic_{idx:05d}"
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{name}.jpg"), bg)
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{name}.txt"), 'w') as f:
            f.write("\n".join(labels))
        idx+=1
        if idx%50==0:
            print(f"Generated {idx}/{NUM_IMAGES}, min count={min(counts.values())}")
    print("Dataset generation completo.")
    print("Counts finali per classe:", counts)

if __name__ == "__main__":
    main()

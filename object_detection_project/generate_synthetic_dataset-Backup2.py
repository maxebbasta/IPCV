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
BACKGROUNDS_DIR     = "dataset/backgrounds"   # sfondi (1.jpg–25.jpg)
MODELS_DIR          = "models"                # ritagli modello (0.jpg,…,23.jpg)
OUTPUT_IMAGES_DIR   = "dataset/images/train"  # dove salvare immagini sintetiche
OUTPUT_LABELS_DIR   = "dataset/labels/train"  # dove salvare etichette YOLO
NUM_IMAGES          = 30000                    # numero minimo di immagini da generare
MIN_OBJS, MAX_OBJS  = 40, 60                   # min/max oggetti per immagine
MIN_OCC_PER_CLASS   = 35000                    # occorrenze minime per ogni classe
MAX_IOU             = 0.15                     # soglia IoU massima per evitare sovrapposizioni
MIN_FACTOR, MAX_FACTOR = 4.5, 6.0              # model height = bg_height / factor
# ------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    xx1, yy1, xx2, yy2 = box2
    ix1, iy1 = max(x1, xx1), max(y1, yy1)
    ix2, iy2 = min(x2, xx2), min(y2, yy2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    area1 = (x2-x1)*(y2-y1)
    area2 = (xx2-xx1)*(yy2-yy1)
    union = area1 + area2 - inter
    return inter/union if union>0 else 0.0


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
        # fornisce canale alpha se manca
        if img.ndim==3 and img.shape[2]==3:
            b,g,r = cv2.split(img)
            alpha = np.full_like(b,255)
            img = cv2.merge((b,g,r,alpha))
        elif not(img.ndim==3 and img.shape[2]==4):
            continue
        models[cls] = img
    if not models:
        raise RuntimeError(f"No model images found in {models_dir}")
    return models


def overlay_image(bg, fg, x, y):
    h_fg, w_fg = fg.shape[:2]
    h_bg, w_bg = bg.shape[:2]
    x1, y1 = max(0,x), max(0,y)
    x2, y2 = min(w_bg, x+w_fg), min(h_bg, y+h_fg)
    fx1, fy1 = x1-x, y1-y
    fx2, fy2 = fx1+(x2-x1), fy1+(y2-y1)
    alpha = fg[fy1:fy2, fx1:fx2, 3:]/255.0
    fg_rgb = fg[fy1:fy2, fx1:fx2, :3]
    roi = bg[y1:y2, x1:x2]
    bg[y1:y2, x1:x2] = (alpha*fg_rgb + (1-alpha)*roi).astype(np.uint8)
    return x1, y1, x2, y2


def rotate_full(fg, angle):
    h,w = fg.shape[:2]
    theta = np.deg2rad(angle)
    cos_t, sin_t = abs(np.cos(theta)), abs(np.sin(theta))
    new_w = int(w*cos_t + h*sin_t)
    new_h = int(w*sin_t + h*cos_t)
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    M[0,2] += (new_w-w)/2
    M[1,2] += (new_h-h)/2
    return cv2.warpAffine(fg, M, (new_w,new_h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))


def random_perspective(img, max_warp=0.2):
    h,w = img.shape[:2]
    def pts(delta):
        return [
            (random.uniform(-delta,delta)*w, random.uniform(-delta,delta)*h),
            (w+random.uniform(-delta,delta)*w, random.uniform(-delta,delta)*h),
            (w+random.uniform(-delta,delta)*w, h+random.uniform(-delta,delta)*h),
            (random.uniform(-delta,delta)*w, h+random.uniform(-delta,delta)*h)
        ]
    src = np.float32([(0,0),(w,0),(w,h),(0,h)])
    dst = np.float32(pts(max_warp))
    M = cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img, M, (w,h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))


def main():
    ensure_dir(OUTPUT_IMAGES_DIR)
    ensure_dir(OUTPUT_LABELS_DIR)

    bg_files = [f for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.jpg','.png'))]
    if not bg_files:
        raise RuntimeError(f"No background images in {BACKGROUNDS_DIR}")
    models = load_models(MODELS_DIR)

    counts = {cls:0 for cls in models.keys()}
    idx = 0
    while idx<NUM_IMAGES or any(counts[c]<MIN_OCC_PER_CLASS for c in counts):
        bg = cv2.imread(os.path.join(BACKGROUNDS_DIR, random.choice(bg_files)))
        if bg is None: continue
        h_bg, w_bg = bg.shape[:2]
        labels, bboxes = [], []
        num_objs = random.randint(MIN_OBJS,MAX_OBJS)
        for _ in range(num_objs):
            cls, fg_orig = random.choice(list(models.items()))
            fh,fw = fg_orig.shape[:2]
            # scala altezza
            factor = random.uniform(MIN_FACTOR,MAX_FACTOR)
            target_h = int(h_bg/factor)
            scale = target_h/float(fh)
            fg = cv2.resize(fg_orig,(int(fw*scale),target_h), interpolation=cv2.INTER_AREA)

            # estrai canali
            bgr = fg[...,:3]
            alpha_ch = fg[...,3:]
            # color jitter
            a = random.uniform(0.8,1.2); b = random.uniform(-30,30)
            bgr = cv2.convertScaleAbs(bgr,alpha=a,beta=b)
            hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[...,1] *= random.uniform(0.7,1.3)
            hsv[...,1] = np.clip(hsv[...,1],0,255)
            bgr = cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
            fg = np.dstack((bgr,alpha_ch))

            # rotazione
            fg = rotate_full(fg, random.uniform(-20,20))

            # occlusione bordo (edge-cut)
            if random.random()<0.5:
                h_fg,w_fg = fg.shape[:2]
                side = random.choice([0,1,2,3])
                depth = int((h_fg if side in [0,2] else w_fg)*random.uniform(0.05,0.70))
                if side==0: fg[:depth,:,3]=0
                elif side==2: fg[-depth:,:,3]=0
                elif side==1: fg[:,-depth:,3]=0
                else: fg[:,:depth,3]=0
            # blur
            if random.random()<0.3:
                k = random.choice([3,5])
                bgr = cv2.GaussianBlur(fg[...,:3],(k,k),0)
                fg = np.dstack((bgr,fg[...,3:]))
            # noise
            if random.random()<0.2:
                noise = np.random.normal(0,10,fg[...,:3].shape).astype(np.int16)
                bgr = np.clip(fg[...,:3].astype(np.int16)+noise,0,255).astype(np.uint8)
                fg = np.dstack((bgr,fg[...,3:]))

            # prospettiva
            if random.random()<0.3:
                fg = random_perspective(fg)

            h_fg,w_fg = fg.shape[:2]
            for _ in range(10):
                x = random.randint(0,max(0,w_bg-w_fg))
                y = random.randint(0,max(0,h_bg-h_fg))
                box = (x,y,x+w_fg,y+y+h_fg)
                if all(compute_iou(box,bb)<MAX_IOU for bb in bboxes):
                    x1,y1,x2,y2 = overlay_image(bg,fg,x,y)
                    bboxes.append((x1,y1,x2,y2))
                    counts[cls]+=1
                    cx = ((x1+x2)/2)/w_bg; cy = ((y1+y2)/2)/h_bg
                    bw = (x2-x1)/w_bg; bh = (y2-y1)/h_bg
                    labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    break

        name = f"synthetic_{idx:05d}"

        # forza altezza = 640px su tutte le immagini
        h_bg, w_bg = bg.shape[:2]
        new_h = 640
        new_w = int(w_bg * new_h / h_bg)
        bg = cv2.resize(bg, (new_w, new_h), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{name}.jpg"), bg)
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{name}.txt"), 'w') as f:
            f.write("\n".join(labels))
        idx+=1
        if idx%50==0:
            print(f"Generated {idx}/{NUM_IMAGES}, min count={min(counts.values())}")

    print("Dataset generation complete.")
    print("Final counts per class:",counts)

if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""
Synthetic Dataset Generator per YOLOv5 con partial/full-height placement.

– Oggetti possono apparire per il 10 % nel frame (anche spigoli tagliati)
– Modalità a random: full-height / partial-edge / normale
– Calcolo corretto delle bbox clippate
"""
import os
import cv2
import numpy as np
import random

# ------------------------
# Configurazione
# ------------------------
BACKGROUNDS_DIR     = "dataset/backgrounds"    # sfondi (1.jpg–25.jpg)
MODELS_DIR          = "models"                 # ritagli modello (0.jpg,…,23.jpg)
OUTPUT_IMAGES_DIR   = "dataset/images/train"   # output immagini
OUTPUT_LABELS_DIR   = "dataset/labels/train"   # output etichette
NUM_IMAGES          = 10000                    # quante generare
MIN_OBJS, MAX_OBJS  = 10, 15                   # oggetti per immagine
MIN_OCC_PER_CLASS   = 2500                     # occorrenze minime per classe
MAX_IOU             = 0.1                      # IoU massima fra bbox
MIN_FACTOR, MAX_FACTOR = 1.0, 6.0              # altezza oggetto = h_bg/factor
# ------------------------

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_iou(b1, b2):
    x1,y1,x2,y2 = b1; xx1,yy1,xx2,yy2 = b2
    ix1, iy1 = max(x1,xx1), max(y1,yy1)
    ix2, iy2 = min(x2,xx2), min(y2,yy2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw*ih
    a1 = (x2-x1)*(y2-y1); a2 = (xx2-xx1)*(yy2-yy1)
    union = a1 + a2 - inter
    return inter/union if union>0 else 0.0

def load_models(models_dir):
    models = {}
    for fn in os.listdir(models_dir):
        if not fn.lower().endswith(('.jpg','.png')): continue
        base = os.path.splitext(fn)[0].split('_')[0]
        try: cls = int(base)
        except: continue
        img = cv2.imread(os.path.join(models_dir,fn), cv2.IMREAD_UNCHANGED)
        if img is None: continue
        # assicurati canale alpha
        if img.ndim==3 and img.shape[2]==3:
            b,g,r = cv2.split(img)
            a = np.full_like(b,255)
            img = cv2.merge((b,g,r,a))
        elif not(img.ndim==3 and img.shape[2]==4):
            continue
        models[cls] = img
    if not models:
        raise RuntimeError("Nessun modello caricato")
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
    return x1,y1,x2,y2

def rotate_full(fg, angle):
    h,w = fg.shape[:2]
    th = np.deg2rad(angle)
    cos, sin = abs(np.cos(th)), abs(np.sin(th))
    nw, nh = int(w*cos + h*sin), int(w*sin + h*cos)
    M = cv2.getRotationMatrix2D((w/2,h/2), angle, 1.0)
    M[0,2] += (nw-w)/2; M[1,2] += (nh-h)/2
    return cv2.warpAffine(fg, M, (nw,nh),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0,0))

def random_perspective(img, max_warp=0.05):
    h,w = img.shape[:2]
    def pts(d):
        return [
            (random.uniform(-d,d)*w, random.uniform(-d,d)*h),
            (w+random.uniform(-d,d)*w, random.uniform(-d,d)*h),
            (w+random.uniform(-d,d)*w, h+random.uniform(-d,d)*h),
            (random.uniform(-d,d)*w, h+random.uniform(-d,d)*h),
        ]
    src = np.float32([(0,0),(w,0),(w,h),(0,h)])
    dst = np.float32(pts(max_warp))
    M = cv2.getPerspectiveTransform(src,dst)
    return cv2.warpPerspective(img, M, (w,h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0,0,0,0))

def main():
    ensure_dir(OUTPUT_IMAGES_DIR)
    ensure_dir(OUTPUT_LABELS_DIR)

    bgs = [f for f in os.listdir(BACKGROUNDS_DIR) if f.lower().endswith(('.jpg','.png'))]
    models = load_models(MODELS_DIR)
    counts = {c:0 for c in models}
    idx = 0

    while idx < NUM_IMAGES or any(counts[c] < MIN_OCC_PER_CLASS for c in counts):
        bg = cv2.imread(os.path.join(BACKGROUNDS_DIR, random.choice(bgs)))
        if bg is None: continue
        h_bg, w_bg = bg.shape[:2]
        labels, bboxes = [], []
        n_obj = random.randint(MIN_OBJS, MAX_OBJS)

        for _ in range(n_obj):
            cls, fg0 = random.choice(list(models.items()))
            fh, fw = fg0.shape[:2]

            # modalità full/partial/normale
            mode = random.random()
            if mode < 0.2:      # full-height
                factor = random.uniform(1.0,1.2)
            elif mode < 0.5:    # partial-edge
                factor = random.uniform(MIN_FACTOR,MAX_FACTOR)
            else:               # normale
                factor = random.uniform(MIN_FACTOR,MAX_FACTOR)

            # scala
            th = int(h_bg / factor)
            scale = th / fh
            fg = cv2.resize(fg0, (int(fw*scale), th), interpolation=cv2.INTER_AREA)

            # color jitter, blur, noise, ecc.
            bgr, a_ch = fg[...,:3], fg[...,3:]
            alpha = random.uniform(0.8,1.2); beta = random.uniform(-30,30)
            bgr = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)
            hsv = cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[...,1] *= random.uniform(0.7,1.3)
            hsv[...,1] = np.clip(hsv[...,1],0,255)
            bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            fg = np.dstack((bgr, a_ch))

            # rotazione
            fg = rotate_full(fg, random.uniform(-15,15))

            # occlusione bordo
            if random.random()<0.3:
                hf, wf = fg.shape[:2]
                side = random.choice([0,1,2,3])
                depth = int((hf if side in [0,2] else wf)*random.uniform(0.1,0.3))
                if side==0: fg[:depth,:,3]=0
                elif side==2: fg[-depth:,:,3]=0
                elif side==1: fg[:,-depth:,3]=0
                else: fg[:,:depth,3]=0

            # blur/noise/erase/persp
            if random.random()<0.3:
                k=random.choice([3,5])
                bgr=cv2.GaussianBlur(fg[...,:3],(k,k),0)
                fg=np.dstack((bgr,fg[...,3:]))
            if random.random()<0.2:
                noise=np.random.normal(0,10,fg[...,:3].shape).astype(np.int16)
                bgr=np.clip(fg[...,:3].astype(np.int16)+noise,0,255).astype(np.uint8)
                fg=np.dstack((bgr,fg[...,3:]))
            if random.random()<0.2:
                hf,wf=fg.shape[:2]
                ew=random.randint(int(wf*0.1),int(wf*0.3))
                eh=random.randint(int(hf*0.1),int(hf*0.3))
                ex=random.randint(0,wf-ew)
                ey=random.randint(0,hf-eh)
                fg[ey:ey+eh,ex:ex+ew,:3]=128
            if random.random()<0.3:
                fg=random_perspective(fg)

            h_fg, w_fg = fg.shape[:2]
            # posizionamento con x,y potenzialmente fuori
            for __ in range(10):
                x = random.randint(int(-0.9*w_fg), int(w_bg - 0.1*w_fg))
                y = random.randint(int(-0.9*h_fg), int(h_bg - 0.1*h_fg))
                box = (x,y,x+w_fg,y+h_fg)
                if all(compute_iou(box,bb)<MAX_IOU for bb in bboxes):
                    x1,y1,x2,y2 = overlay_image(bg, fg, x, y)
                    if x2-x1 < 2 or y2-y1 < 2:
                        break
                    bboxes.append((x1,y1,x2,y2))
                    counts[cls] += 1
                    cx = ((x1+x2)/2)/w_bg; cy = ((y1+y2)/2)/h_bg
                    bw = (x2-x1)/w_bg; bh = (y2-y1)/h_bg
                    labels.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    break

        name = f"synthetic_{idx:05d}"
        cv2.imwrite(os.path.join(OUTPUT_IMAGES_DIR, f"{name}.jpg"), bg)
        with open(os.path.join(OUTPUT_LABELS_DIR, f"{name}.txt"), 'w') as f:
            f.write("\n".join(labels))

        idx += 1
        if idx % 50 == 0:
            print(f"Generated {idx}/{NUM_IMAGES}, min count={min(counts.values())}")

    print("Dataset generation completo.")
    print("Counts finali per classe:", counts)

if __name__ == "__main__":
    main()

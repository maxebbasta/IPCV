# Step A: Product Recognition su scaffali con SIFT
# + Rotated Rectangle per bounding‑box più preciso (fix per numpy int0)

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# -------------------------------------------------
# 1) Parametri globali
# -------------------------------------------------
MIN_MATCH_COUNT         = 4      # min match per tentare omografia
MIN_INLIERS             = 12     # min inlier RANSAC per validare rilevamento
INLIER_RATIO_THRESH     = 0.2    # inlier/match minimo
MIN_BOX_SIZE            = 20     # min larghezza/altezza in px
RATIO_TEST_THRESHOLD    = 0.75   # soglia ratio test di Lowe
RANSAC_REPROJ_THRESHOLD = 3.0    # soglia reproj per RANSAC

model_ids   = ["0","1","11","19","24","25","26"]
scene_files = [f"e{i}.png" for i in range(1,6)]

# -------------------------------------------------
# 2) Caricamento immagini modello (grayscale)
# -------------------------------------------------
model_images = {}
for mid in model_ids:
    img = cv2.imread(f"models/{mid}.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Modello {mid}.jpg non trovato")
    model_images[mid] = img

# -------------------------------------------------
# 3) Caricamento immagini di scena (color + grayscale)
# -------------------------------------------------
scene_images      = {}
scene_images_gray = {}
for sf in scene_files:
    img_c = cv2.imread(f"scenes/{sf}", cv2.IMREAD_COLOR)
    if img_c is None:
        raise FileNotFoundError(f"Scena {sf} non trovata")
    scene_images[sf]      = img_c
    scene_images_gray[sf] = cv2.cvtColor(img_c, cv2.COLOR_BGR2GRAY)

# -------------------------------------------------
# 4) Inizializza SIFT e BFMatcher (L2)
# -------------------------------------------------
sift  = cv2.SIFT_create()
bf_l2 = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

# -------------------------------------------------
# 5) Estrazione keypoints e descriptors
# -------------------------------------------------
model_keypoints   = {}
model_descriptors = {}
for mid, img in model_images.items():
    kp, desc = sift.detectAndCompute(img, None)
    model_keypoints[mid]   = kp
    model_descriptors[mid] = desc

scene_keypoints   = {}
scene_descriptors = {}
for sf, img_gray in scene_images_gray.items():
    kp, desc = sift.detectAndCompute(img_gray, None)
    scene_keypoints[sf]   = kp
    scene_descriptors[sf] = desc

# -------------------------------------------------
# 6) Matching con ratio test di Lowe
# -------------------------------------------------
good_matches_dict = {sf: {mid: [] for mid in model_ids} for sf in scene_files}

for sf in scene_files:
    desc_scene = scene_descriptors[sf]
    for mid in model_ids:
        desc_model = model_descriptors[mid]
        if desc_model is None or desc_scene is None:
            continue
        raw  = bf_l2.knnMatch(desc_model, desc_scene, k=2)
        good = [m for m,n in raw if m.distance < RATIO_TEST_THRESHOLD * n.distance]
        good = sorted(good, key=lambda x: x.distance)
        good_matches_dict[sf][mid] = good

# -------------------------------------------------
# 7) Omografia + exclusion + rotated rectangle
# -------------------------------------------------
def point_in_box(pt, box):
    x0, y0, w, h = box
    x, y = pt
    return (x0 <= x <= x0+w) and (y0 <= y <= y0+h)

detections   = {sf: {mid: [] for mid in model_ids} for sf in scene_files}
used_regions = {sf: [] for sf in scene_files}

for sf in scene_files:
    kp_scene   = scene_keypoints[sf]

    for mid in model_ids:
        kp_model   = model_keypoints[mid]
        matches    = good_matches_dict[sf][mid]

        # 1) Filtra match in aree già usate
        filt = []
        for m in matches:
            pt = kp_scene[m.trainIdx].pt
            if not any(point_in_box(pt, box) for box in used_regions[sf]):
                filt.append(m)
        matches = filt

        if len(matches) < MIN_MATCH_COUNT:
            continue

        # 2) Prepara punti e stima omografia
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt  for m in matches]).reshape(-1,1,2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)
        if H is None:
            continue
        mask = mask.ravel()
        inliers = int(mask.sum())
        if inliers < MIN_INLIERS or (inliers/len(matches)) < INLIER_RATIO_THRESH:
            continue

        # 3) Proietta corner del modello
        h_m, w_m      = model_images[mid].shape
        corners       = np.float32([[0,0],[w_m,0],[w_m,h_m],[0,h_m]]).reshape(-1,1,2)
        scene_corners = cv2.perspectiveTransform(corners, H)

        # 4) Calcola rotated rectangle
        rect = cv2.minAreaRect(scene_corners)  # ((cx,cy),(w_rect,h_rect),angle)
        box  = cv2.boxPoints(rect)
        box  = box.astype(int)             # <--- qui la correzione
        (cx, cy), (w_rect, h_rect), angle = rect

        # 5) Filtri area minima
        if w_rect < MIN_BOX_SIZE or h_rect < MIN_BOX_SIZE:
            continue

        # 6) Salva detection e marca area
        detections[sf][mid].append({
            "box_pts": box,
            "position": (int(cx), int(cy)),
            "width":    f"{int(w_rect)} px",
            "height":   f"{int(h_rect)} px",
            "angle":    angle
        })
        # usa axis-aligned wrapper per esclusione
        x0, y0, w0, h0 = cv2.boundingRect(box)
        used_regions[sf].append((x0, y0, w0, h0))

# -------------------------------------------------
# 8) Visualizzazione rotated rectangles
# -------------------------------------------------
random.seed(0)
colors = { mid: tuple(map(int, np.random.choice(range(50,256), size=3))) for mid in model_ids }

for sf, img_color in scene_images.items():
    img_draw = img_color.copy()
    for mid in model_ids:
        for inst in detections[sf][mid]:
            box = inst["box_pts"]
            col = colors[mid]
            cv2.drawContours(img_draw, [box], 0, col, 2)
            x0, y0 = box[0]
            cv2.putText(img_draw, f"ID {mid}", (x0, y0-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    plt.figure(figsize=(10,6))
    plt.imshow(cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB))
    plt.title(f"Scene {sf} – Precise Rotated Boxes")
    plt.axis('off')
    plt.show()

# -------------------------------------------------
# 9) Stampa dei risultati
# -------------------------------------------------
for sf in scene_files:
    print(f"\nScene {sf}:")
    for mid in model_ids:
        inst_list = detections[sf][mid]
        print(f"  Model {mid}: {len(inst_list)} instance(s)")
        for i, inst in enumerate(inst_list, start=1):
            cx, cy = inst["position"]
            W, H   = inst["width"], inst["height"]
            ang    = inst["angle"]
            print(f"    Instance {i}: {{position: ({cx}, {cy}), width: {W}, height: {H}, angle: {ang:.1f}}}")

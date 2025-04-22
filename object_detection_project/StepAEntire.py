#!/usr/bin/env python3
# StepAEntire.py
import cv2
import numpy as np

# -------------------------------------------
# Fase A: Rilevamento singole istanze con SIFT,
# Omografia e bounding box ruotato (minAreaRect).
# -------------------------------------------

# Directory dei modelli e delle scene
models_dir = "./models/"    # immagini modello: 0.jpg, 1.jpg, 11.jpg, ...
scenes_dir = "./scenes/"    # immagini di scena: e1.png, e2.png, ...

# ID dei modelli e nomi delle scene
model_ids = [0, 1, 11, 19, 24, 25, 26]
scene_files = ["e1.png", "e2.png", "e3.png", "e4.png", "e5.png"]

# Inizializza SIFT e BFMatcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)

# Pre-elabora i modelli: carica immagine, keypoint, descrittori, dimensioni
models = {}
for mid in model_ids:
    path = f"{models_dir}{mid}.jpg"
    img_model = cv2.imread(path)
    if img_model is None:
        raise FileNotFoundError(f"Model image not found: {path}")
    kp_model, des_model = sift.detectAndCompute(img_model, None)
    h_model, w_model = img_model.shape[:2]
    models[mid] = {
        'kp': kp_model,
        'des': des_model,
        'size': (w_model, h_model)
    }

# Processa ciascuna immagine di scena
for scene_file in scene_files:
    scene_path = f"{scenes_dir}{scene_file}"
    img_scene = cv2.imread(scene_path)
    if img_scene is None:
        raise FileNotFoundError(f"Scene image not found: {scene_path}")
    h_scene, w_scene = img_scene.shape[:2]

    # Estrai keypoint e descrittori SIFT dalla scena
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)

    detections = {}
    # Per ogni modello, matching e omografia
    for mid, data in models.items():
        kp_model = data['kp']
        des_model = data['des']
        w_model, h_model = data['size']
        if des_model is None or des_scene is None:
            continue

        # 1) Matching SIFT + Lowe ratio test
        matches = bf.knnMatch(des_model, des_scene, k=2)
        good = [m for m, n in matches if m.distance < 0.4 * n.distance]
        if len(good) < 10:
            continue

        # 2) Stima omografia con RANSAC
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if M is None:
            continue
        inliers = int(mask.sum())
        if inliers < 4:
            continue

        # 3) Trasforma corner del modello
        corners = np.float32([[0,0],[w_model,0],[w_model,h_model],[0,h_model]]).reshape(-1,1,2)
        dst_c = cv2.perspectiveTransform(corners, M)
        pts = dst_c.reshape(-1,2).astype(np.float32)

        # 4) Calcola bbox ruotato via minAreaRect
        rot_rect = cv2.minAreaRect(pts)
        (cx, cy), (w_box, h_box), angle = rot_rect
        box_pts = cv2.boxPoints(rot_rect).astype(int)

        # 5) Filtro area minima (almeno 1% area scena)
        if w_box * h_box < 0.01 * (w_scene * h_scene):
            continue

        # Salva detection
        detections.setdefault(mid, []).append({
            'center': (int(round(cx)), int(round(cy))),
            'width': int(round(w_box)),
            'height': int(round(h_box)),
            'angle': angle,
            'box_pts': box_pts
        })

    # Stampa risultati
    print(f"\nRisultati per {scene_file}:")
    if not detections:
        print("  Nessun prodotto riconosciuto.")
    for pid, dets in detections.items():
        print(f"  Product {pid} - {len(dets)} instance(s) found:")
        for idx, det in enumerate(dets, 1):
            cx, cy = det['center']
            w_box, h_box = det['width'], det['height']
            print(f"    Instance {idx} {{position: ({cx},{cy}), width: {w_box}px, height: {h_box}px}}")

    # Disegna e visualizza
    for dets in detections.values():
        for det in dets:
            cv2.drawContours(img_scene, [det['box_pts']], 0, (0,255,0), 2)
            cv2.circle(img_scene, det['center'], 4, (0,0,255), -1)

    cv2.imshow(f"Detections - {scene_file}", img_scene)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

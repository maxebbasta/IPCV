#!/usr/bin/env python3
import cv2
import numpy as np
import os

# -------------------------------------------
# Fase A migliorata: Rilevamento con SIFT,
# Omografia, debug e bounding box ruotato
# -------------------------------------------

# Configurazione
MODELS_DIR   = "./models/"    # immagini modello: 0.jpg, 1.jpg, ...
SCENES_DIR   = "./scenes/"    # immagini scena: e1.png, e2.png, ...
MODEL_IDS    = [0, 1, 11, 19, 24, 25, 26]
SCENE_FILES  = ["e1.png", "e2.png", "e3.png", "e4.png", "e5.png"]

# Parametri di matching e filtro
MIN_MATCHES     = 35     # numero minimo di match valida/o e di inliers
RATIO_TEST      = 0.7    # soglia Lowe
MIN_AREA_RATIO  = 0.1   # area minima relativa alla scena
RANSAC_THRESH   = 3    # soglia RANSAC in pixel

# Funzione di pre-processing scena (CLAHE)
def preprocess_scene(img_scene):
    lab = cv2.cvtColor(img_scene, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Inizializza SIFT e BFMatcher
sift = cv2.SIFT_create()
bf = cv2.BFMatcher(cv2.NORM_L2)

# Caricamento e pre-elaborazione dei modelli
models = {}
for mid in MODEL_IDS:
    path = os.path.join(MODELS_DIR, f"{mid}.jpg")
    img_model = cv2.imread(path)
    if img_model is None:
        raise FileNotFoundError(f"Model image not found: {path}")
    kp_model, des_model = sift.detectAndCompute(img_model, None)
    h_model, w_model = img_model.shape[:2]
    models[mid] = {
        'img': img_model,
        'kp': kp_model,
        'des': des_model,
        'size': (w_model, h_model)
    }

# Processa ciascuna immagine di scena
for scene_file in SCENE_FILES:
    scene_path = os.path.join(SCENES_DIR, scene_file)
    img_scene = cv2.imread(scene_path)
    if img_scene is None:
        raise FileNotFoundError(f"Scene image not found: {scene_path}")

    # Pre-process della scena per miglior contrasto
    img_proc = preprocess_scene(img_scene)
    h_scene, w_scene = img_proc.shape[:2]

    # Estrai keypoint e descrittori dalla scena
    kp_scene, des_scene = sift.detectAndCompute(img_proc, None)
    print(f"\n=== Analisi scena {scene_file} ===")
    print(f"Kp scena: {len(kp_scene)}  Descriptors scena: {None if des_scene is None else des_scene.shape}")

    detections = {}
    # Matching e omografia per ogni modello
    for mid, data in models.items():
        kp_model = data['kp']
        des_model = data['des']
        w_model, h_model = data['size']

        print(f"\n[Modello {mid}]: kp_modello={len(kp_model)}  des_modello={None if des_model is None else des_model.shape}")
        if des_model is None or des_scene is None:
            print("  => Skip: descrittori assenti")
            continue

        # 1) Matching SIFT + Lowe ratio test
        matches = bf.knnMatch(des_model, des_scene, k=2)
        print(f"  Raw matches: {len(matches)}")
        good = [m for m,n in matches if m.distance < RATIO_TEST * n.distance]
        print(f"  Good matches (ratio<{RATIO_TEST}): {len(good)}")
        if len(good) < MIN_MATCHES:
            print(f"  => Skip: too few good matches (<{MIN_MATCHES})")
            continue

        # 2) Stima omografia con RANSAC
        src_pts = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
        if M is None:
            print("  => Skip: omografia non trovata")
            continue
        inliers = int(mask.sum())
        print(f"  Inliers RANSAC: {inliers}")
        if inliers < MIN_MATCHES:
            print(f"  => Skip: pochi inliers (<{MIN_MATCHES})")
            continue

        # 3) Trasforma corner del modello
        corners = np.float32([[0,0],[w_model,0],[w_model,h_model],[0,h_model]]).reshape(-1,1,2)
        dst_c = cv2.perspectiveTransform(corners, M)
        pts = dst_c.reshape(-1,2).astype(np.float32)

        # 4) Calcola bbox ruotato via minAreaRect
        rot_rect = cv2.minAreaRect(pts)
        (cx, cy), (w_box, h_box), angle = rot_rect
        box_pts = cv2.boxPoints(rot_rect).astype(int)

        # 5) Filtro area minima
        if w_box * h_box < MIN_AREA_RATIO * (w_scene * h_scene):
            print(f"  => Skip: area troppo piccola (<{MIN_AREA_RATIO*100:.2f}% scena)")
            continue

        # Salva detection
        detections.setdefault(mid, []).append({
            'center': (int(round(cx)), int(round(cy))),
            'width': int(round(w_box)),
            'height': int(round(h_box)),
            'angle': angle,
            'box_pts': box_pts
        })
        print(f"  *** Rilevata istanza modello {mid} ({inliers} inliers) ***")

    # Report risultati
    print(f"\nRisultati per {scene_file}:")
    if not detections:
        print("  Nessun prodotto riconosciuto.")
    for pid, dets in detections.items():
        print(f"  Modello {pid} - {len(dets)} istanza(e) individuata(e):")
        for idx, det in enumerate(dets, 1):
            cx, cy = det['center']
            w_box, h_box = det['width'], det['height']
            print(f"    Istanza {idx} {{posizione: ({cx},{cy}), w={w_box}px, h={h_box}px, angolo={det['angle']:.1f}°}}")

    # Disegna e mostra risultati
    vis = img_scene.copy()
    for dets in detections.values():
        for det in dets:
            cv2.drawContours(vis, [det['box_pts']], 0, (0,255,0), 2)
            cv2.circle(vis, det['center'], 4, (0,0,255), -1)
    cv2.imshow(f"Detections - {scene_file}", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
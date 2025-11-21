import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

MODELS_DIR = "./models/"
SCENES_DIR = "./scenes/"

MODEL_IDS = [0, 1, 11, 19, 24, 25, 26]
SCENE_FILES = ["e1.png", "e2.png", "e3.png", "e4.png", "e5.png"]

MIN_MATCHES = 20
RATIO_TEST = 0.7
RANSAC_THRESH = 5.0

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

# Carico i modelli
models = {}
for mid in MODEL_IDS:
    img = cv2.imread(os.path.join(MODELS_DIR, f"{mid}.jpg"))
    if img is None:
        continue
    
    kp, des = sift.detectAndCompute(img, None)
    h, w = img.shape[:2]
    
    models[mid] = {
        "img": img,
        "kp": kp,
        "des": des,
        "size": (w, h)
    }

# Processamento delle scene
for scene_file in SCENE_FILES:
    scene = cv2.imread(os.path.join(SCENES_DIR, scene_file))
    kp_s, des_s = sift.detectAndCompute(scene, None)

    detections = {}

    for mid, mdata in models.items():
        kp_m = mdata["kp"]
        des_m = mdata["des"]

        # Matching + ratio test (come negli esercizi del prof)
        matches = bf.knnMatch(des_m, des_s, k=2)
        good = []
        for m, n in matches:
            if m.distance < RATIO_TEST * n.distance:
                good.append(m)

        if len(good) < MIN_MATCHES:
            continue

        # Omografia (esattamente come nel lab)
        src = np.float32([kp_m[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_THRESH)
        if H is None:
            continue

        # Trasformazione angoli del modello → scena
        w, h = mdata["size"]
        box = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
        proj = cv2.perspectiveTransform(box, H)

        detections[mid] = proj

    # Visualizzazione come nel lab del prof
    vis = scene.copy()
    for mid, proj in detections.items():
        cv2.polylines(vis, [np.int32(proj)], True, (0,255,0), 3)
        center = proj.mean(axis=0).ravel()
        cv2.putText(vis, str(mid), (int(center[0]), int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    plt.figure(figsize=(10,7))
    plt.title(scene_file)
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
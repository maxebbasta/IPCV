import cv2
import numpy as np

# Parametri
models_dir = "./models/"   # directory contenente le immagini modello
scenes_dir = "./scenes/"   # directory contenente le immagini di scena
model_ids = [0, 1, 11, 19, 24, 25, 26]
scene_files = ["e1.png", "e2.png", "e3.png", "e4.png", "e5.png"]

# Inizializza SIFT
sift = cv2.SIFT_create()

# Pre-calcola keypoint e descrittori per ogni modello
models_features = {}
for mid in model_ids:
    img_model = cv2.imread(models_dir + f"{mid}.jpg")
    if img_model is None:
        raise FileNotFoundError(f"Model {mid}.jpg non trovato in {models_dir}")
    kp_model, des_model = sift.detectAndCompute(img_model, None)
    h_model, w_model = img_model.shape[:2]
    models_features[mid] = (kp_model, des_model, h_model, w_model, img_model)

# Matcher brute-force
bf = cv2.BFMatcher(cv2.NORM_L2)

for scene_file in scene_files:
    img_scene = cv2.imread(scenes_dir + scene_file)
    if img_scene is None:
        raise FileNotFoundError(f"Scene {scene_file} non trovato in {scenes_dir}")
    h_scene, w_scene = img_scene.shape[:2]
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)

    detected_products = {}
    for mid, (kp_model, des_model, h_model, w_model, img_model) in models_features.items():
        if des_model is None or des_scene is None:
            continue
        # Ratio-test di Lowe
        matches = bf.knnMatch(des_model, des_scene, k=2)
        good = [m for m,n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            continue
        # Estrazione punti
        src = np.float32([kp_model[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst = np.float32([kp_scene[m.trainIdx].pt for m in good]).reshape(-1,1,2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        inliers = int(mask.sum()) if mask is not None else 0
        # Verifica numero minimo di inlier
        if M is None or inliers < 10 or inliers / len(good) < 0.3:
            continue
        # Calcolo fattore di scala dall'omografia
        A = M[:2, :2]
        scale = np.sqrt(abs(np.linalg.det(A)))
        # Applica omografia ai corner del modello
        corners = np.float32([[0,0],[w_model,0],[w_model,h_model],[0,h_model]]).reshape(-1,1,2)
        dst_c = cv2.perspectiveTransform(corners, M)
        xs = dst_c[:,0,0]; ys = dst_c[:,0,1]
        # Coordinate non filtrate
        x_min = xs.min(); x_max = xs.max()
        y_min = ys.min(); y_max = ys.max()
        # Clamping ai bordi dell'immagine
        x_min_c = max(0, min(w_scene-1, int(round(x_min))))
        y_min_c = max(0, min(h_scene-1, int(round(y_min))))
        x_max_c = max(0, min(w_scene-1, int(round(x_max))))
        y_max_c = max(0, min(h_scene-1, int(round(y_max))))
        w = x_max_c - x_min_c
        h = y_max_c - y_min_c
        # Filtra in base all'area prevista dalla scala
        area_model = w_model * h_model
        area_box = w * h
        expected_area = (scale**2) * area_model
        if not (0.5 * expected_area <= area_box <= 2.0 * expected_area):
            continue
        # Calcola centro
        cx = x_min_c + w // 2
        cy = y_min_c + h // 2

        detected_products.setdefault(mid, []).append((x_min_c, y_min_c, w, h, cx, cy))

    # Stampa i risultati
    print(f"Risultati per {scene_file}:")
    if not detected_products:
        print("  Nessun prodotto riconosciuto.")
    for pid, inst_list in detected_products.items():
        print(f"  Product {pid} - {len(inst_list)} instance(s) found:")
        for i, (x,y,w,h,cx,cy) in enumerate(inst_list,1):
            print(f"    Instance {i} {{position: ({cx},{cy}), width: {w}px, height: {h}px}}")
            # Disegna bounding box
            cv2.rectangle(img_scene, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(img_scene, f"{pid}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # Visualizza
    cv2.imshow(f"Detections - {scene_file}", img_scene)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

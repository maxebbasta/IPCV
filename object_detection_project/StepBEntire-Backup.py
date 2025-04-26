#!/usr/bin/env python3
import cv2
import numpy as np
import os

# -------------------------------------------
# Fase B: Rilevamento di istanze multiple con SIFT + GHT
# -------------------------------------------

# Configurazione
MODELS_DIR       = "./models/"    # cartella immagini modello
SCENES_DIR       = "./scenes/"    # cartella immagini scena
MODEL_IDS        = [0, 1, 11, 19, 24, 25, 26]  # ID modelli (nomi file .jpg)
SCENE_FILES      = ["m1.png", "m2.png", "m3.png", "m4.png", "m5.png"]  # file immagini scena

# Modelli da discriminare con filtro colore (hue)
CONFUSE_MODELS   = {1, 11, 0, 26}
HUE_DIFF_THRESH  = 18  # differenza massima Hue ammessa (gradi)

# Parametri di matching e filtro
MIN_MATCHES         = 30     # numero minimo di match totali per considerare il modello
RATIO_TEST          = 0.7    # soglia Lowe per filtro ratio test
RANSAC_THRESH       = 3.0    # soglia RANSAC in pixel per omografia
MIN_AREA_RATIO      = 0.01   # area minima rilevata rispetto all'area scena (filtro)
CLUSTER_EPS         = 30.0   # raggio (in pixel) per clustering dei voti GHT
MIN_CLUSTER_INLIERS = 10     # numero minimo di inliers richiesti per accettare una istanza

# Stile disegno risultati
BOX_COLOR    = (0, 255, 0)   # colore verde per bounding box
CENTER_COLOR = (0, 0, 255)   # colore rosso per centroide
TEXT_COLOR   = (0, 255, 0)   # colore testo (verde)
FONT         = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE   = 0.7
TEXT_THICK   = 2
BOX_THICK    = 10
CIRCLE_RAD   = 4

# Funzione di pre-processing scena (es. applica CLAHE sul canale L per migliorare contrasto)
def preprocess_scene(img_scene):
    lab = cv2.cvtColor(img_scene, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    lab = cv2.merge((cl, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

# Calcola il valore medio di Hue di un'immagine BGR
def calc_mean_hue(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 0]))

# Inizializza SIFT e BFMatcher
sift = cv2.SIFT_create()
bf   = cv2.BFMatcher(cv2.NORM_L2)

# Caricamento e pre-elaborazione dei modelli
models = {}
for mid in MODEL_IDS:
    path = os.path.join(MODELS_DIR, f"{mid}.jpg")
    img_model = cv2.imread(path)
    if img_model is None:
        raise FileNotFoundError(f"Model image not found: {path}")
    # Estrai keypoints e descrittori del modello
    kp_model, des_model = sift.detectAndCompute(img_model, None)
    h_model, w_model = img_model.shape[:2]
    # Calcola centroide (barycenter) dei keypoints del modello
    if kp_model is not None and len(kp_model) > 0:
        pts = np.float32([kp.pt for kp in kp_model])
        cx_model = float(np.mean(pts[:,0]))
        cy_model = float(np.mean(pts[:,1]))
    else:
        # se non ci sono keypoint, usa centro dell'immagine come fallback
        cx_model = w_model / 2.0
        cy_model = h_model / 2.0
    # Calcola mean hue per modelli "confusi" (da filtrare via colore)
    mean_hue = None
    if mid in CONFUSE_MODELS:
        mean_hue = calc_mean_hue(img_model)
    # Salva dati del modello
    models[mid] = {
        'img': img_model,
        'kp': kp_model,
        'des': des_model,
        'size': (w_model, h_model),
        'centroid': (cx_model, cy_model),
        'mean_hue': mean_hue
    }

# Processa ciascuna immagine di scena
for scene_file in SCENE_FILES:
    scene_path = os.path.join(SCENES_DIR, scene_file)
    img_scene = cv2.imread(scene_path)
    if img_scene is None:
        raise FileNotFoundError(f"Scene image not found: {scene_path}")
    # Pre-processing della scena
    img_proc = preprocess_scene(img_scene)
    h_scene, w_scene = img_proc.shape[:2]
    # Estrai keypoints e descrittori della scena
    kp_scene, des_scene = sift.detectAndCompute(img_proc, None)
    print(f"\n=== Analisi scena {scene_file} ===")
    print(f"Kp scena: {len(kp_scene)}  Descriptors: {None if des_scene is None else des_scene.shape}")
    # Dizionario per detections trovate nella scena corrente
    detections = {}
    # Per ogni modello, esegui matching e Generalized Hough Transform per rilevare istanze
    for mid, data in models.items():
        kp_model = data['kp']
        des_model = data['des']
        w_model, h_model = data['size']
        cx_model, cy_model = data['centroid']
        print(f"\n[Modello {mid}]: kp_modello={len(kp_model)}  des_modello={None if des_model is None else des_model.shape}")
        if des_model is None or des_scene is None:
            print("  => Skip: descrittori assenti")
            continue
        # 1) Matching SIFT con Lowe's ratio test
        matches = bf.knnMatch(des_model, des_scene, k=2)
        print(f"  Raw matches: {len(matches)}")
        good = [m for m, n in matches if m.distance < RATIO_TEST * n.distance]
        print(f"  Good matches (ratio<{RATIO_TEST}): {len(good)}")
        if len(good) < MIN_MATCHES:
            print(f"  => Skip: too few good matches (<{MIN_MATCHES})")
            continue
        # 2) GHT: voto per posizione del centroide del modello nella scena per ogni match
        votes = []  # lista dei voti (posizioni predette del centroide nella scena)
        for m in good:
            # Coordinate keypoint modello e scena
            kp_m = kp_model[m.queryIdx]
            kp_s = kp_scene[m.trainIdx]
            (mx, my) = kp_m.pt
            (sx, sy) = kp_s.pt
            # Vettore dal keypoint modello al centroide del modello
            v_mx = cx_model - mx
            v_my = cy_model - my
            # Calcola differenza di orientazione e scala relativa
            angle_m = kp_m.angle
            angle_s = kp_s.angle
            angle_diff = angle_s - angle_m
            # Usa coseno e seno dell'angolo per ruotare il vettore
            angle_diff_rad = np.deg2rad(angle_diff)
            cosA = np.cos(angle_diff_rad)
            sinA = np.sin(angle_diff_rad)
            # Vettore ruotato secondo la differenza di orientazione
            v_rot_x = v_mx * cosA - v_my * sinA
            v_rot_y = v_mx * sinA + v_my * cosA
            # Scala il vettore in base al rapporto di scala dei keypoint
            scale_m = kp_m.size
            scale_s = kp_s.size
            scale_ratio = scale_s / scale_m if scale_m > 1e-6 else 1.0
            v_rot_scaled_x = v_rot_x * scale_ratio
            v_rot_scaled_y = v_rot_y * scale_ratio
            # Posizione votata del centroide nella scena (trasla il keypoint di scena di questo vettore)
            cx_pred = sx + v_rot_scaled_x
            cy_pred = sy + v_rot_scaled_y
            votes.append((cx_pred, cy_pred))
        # 3) Clustering dei voti per individuare istanze multiple dello stesso prodotto
        vote_count = len(votes)
        cluster_indices_list = []
        visited = [False] * vote_count
        for i in range(vote_count):
            if visited[i]:
                continue
            # Avvia un nuovo cluster con il voto i
            visited[i] = True
            cluster_idx = [i]
            # BFS/DFS per trovare tutti i voti vicini entro CLUSTER_EPS
            queue = [i]
            while queue:
                cur = queue.pop(0)
                (cx_cur, cy_cur) = votes[cur]
                for j in range(vote_count):
                    if not visited[j]:
                        (cx_j, cy_j) = votes[j]
                        dx = cx_cur - cx_j
                        dy = cy_cur - cy_j
                        if dx*dx + dy*dy <= CLUSTER_EPS * CLUSTER_EPS:
                            visited[j] = True
                            cluster_idx.append(j)
                            queue.append(j)
            cluster_indices_list.append(cluster_idx)
        # Per ogni cluster di voti trovato, stima l'istanza del prodotto
        for ci, cluster_idx in enumerate(cluster_indices_list, start=1):
            cluster_size = len(cluster_idx)
            if cluster_size < 4:
                print(f"  => Skip: cluster {ci} troppo piccolo ({cluster_size} punti)")
                continue
            # 4) Stima omografia con RANSAC per il cluster corrente
            src_pts = np.float32([kp_model[good[idx].queryIdx].pt for idx in cluster_idx]).reshape(-1,1,2)
            dst_pts = np.float32([kp_scene[good[idx].trainIdx].pt for idx in cluster_idx]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESH)
            if M is None:
                print(f"  => Skip: omografia non trovata per cluster {ci}")
                continue
            inliers = int(mask.sum())
            print(f"  Cluster {ci}: {cluster_size} match, Inliers RANSAC = {inliers}")
            if inliers < MIN_CLUSTER_INLIERS:
                print(f"  => Skip: pochi inliers per cluster {ci} (<{MIN_CLUSTER_INLIERS})")
                continue
            # Trasforma i quattro angoli dell'immagine modello secondo l'omografia trovata
            corners = np.float32([[0,0], [w_model,0], [w_model,h_model], [0,h_model]]).reshape(-1,1,2)
            dst_corners = cv2.perspectiveTransform(corners, M)
            pts = dst_corners.reshape(-1, 2).astype(np.float32)
            # 5) Calcola bounding box ruotato dell'istanza
            rot_rect = cv2.minAreaRect(pts)
            (cx, cy), (w_box, h_box), angle = rot_rect
            box_pts = cv2.boxPoints(rot_rect).astype(int)
            # Filtro area minima
            if w_box * h_box < MIN_AREA_RATIO * (w_scene * h_scene):
                print(f"  => Skip: area cluster {ci} troppo piccola (<{MIN_AREA_RATIO*100:.2f}% scena)")
                continue
            # 6) Filtro colore (se necessario per modelli confondibili)
            if mid in CONFUSE_MODELS:
                src_rect = np.float32([[0,0], [w_model,0], [w_model,h_model], [0,h_model]])
                dst_rect = np.float32(box_pts)
                M_inv = cv2.getPerspectiveTransform(dst_rect, src_rect)
                roi = cv2.warpPerspective(img_scene, M_inv, (w_model, h_model))
                mean_h_roi = calc_mean_hue(roi)
                diff_h = abs(mean_h_roi - data['mean_hue'])
                print(f"  Hue ROI={mean_h_roi:.1f}, Modello={data['mean_hue']:.1f}, Δ={diff_h:.1f}")
                if diff_h > HUE_DIFF_THRESH:
                    print(f"  => Skip: differenza colore cluster {ci} troppo alta (>±{HUE_DIFF_THRESH}°)")
                    continue
            # 7) Salva detection trovata per questo cluster/istanza
            detections.setdefault(mid, []).append({
                'center': (int(round(cx)), int(round(cy))),
                'width':  int(round(w_box)),
                'height': int(round(h_box)),
                'angle':  angle,
                'box_pts': box_pts
            })
            print(f"  *** Rilevata istanza modello {mid} (cluster {ci}, {inliers} inliers) ***")
    # Report dei risultati per l'immagine di scena corrente
    print(f"\nRisultati per {scene_file}:")
    if not detections:
        print("  Nessun prodotto riconosciuto.")
    for pid, dets in detections.items():
        print(f"  Modello {pid} - {len(dets)} istanza(e) individuata(e):")
        for idx, det in enumerate(dets, start=1):
            cx, cy = det['center']
            w_b, h_b = det['width'], det['height']
            print(f"    Istanza {idx} {{posizione: ({cx},{cy}), w={w_b}px, h={h_b}px}}")
    # Disegna e mostra i risultati sulla scena
    vis = img_scene.copy()
    for pid, dets in detections.items():
        for det in dets:
            # Disegna il bounding box ruotato
            cv2.drawContours(vis, [det['box_pts']], 0, BOX_COLOR, BOX_THICK)
            # Disegna il centroide
            cv2.circle(vis, det['center'], CIRCLE_RAD, CENTER_COLOR, -1)
            # Stampa l'ID del prodotto centrato rispetto al box
            text = f"ID: {pid}"
            (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, TEXT_THICK)
            tx = det['center'][0] - tw // 2
            ty = det['center'][1] + th // 2
            cv2.putText(vis, text, (tx, ty), FONT, FONT_SCALE, TEXT_COLOR, TEXT_THICK)
    cv2.imshow(f"Detections - {scene_file}", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

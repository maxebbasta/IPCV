import cv2
import numpy as np
import math

# Parametri
models_dir = "./models/"
scenes_dir = "./scenes/"
model_ids = [0, 1, 11, 19, 24, 25, 26]  # stessi modelli della fase A
scene_files = ["m1.png", "m2.png", "m3.png", "m4.png", "m5.png"]  # scene con istanze multiple

# Inizializza SIFT
sift = cv2.SIFT_create()

# Pre-elabora i modelli: estrai feature SIFT e calcola baricentro delle keypoint di ciascun modello
models_data = {}
for mid in model_ids:
    img_model = cv2.imread(models_dir + f"{mid}.jpg")
    if img_model is None:
        raise FileNotFoundError(f"Model image {mid}.jpg not found")
    kp_model, des_model = sift.detectAndCompute(img_model, None)
    if kp_model is None or len(kp_model) == 0:
        continue
    # Calcola baricentro (barycenter) delle keypoint del modello
    pts = np.array([kp.pt for kp in kp_model], dtype=np.float32)
    barycenter = tuple(np.mean(pts, axis=0))  # (x_b, y_b)
    models_data[mid] = {
        "kp": kp_model,
        "des": des_model,
        "size": img_model.shape[:2],  # (h, w)
        "barycenter": barycenter
    }

# Funzione di clustering dei voti (predicted center points) per trovare picchi (istanze)
def cluster_points(points, dist_threshold):
    """Clusterizzazione semplice tipo DBSCAN: raggruppa punti vicini entro dist_threshold."""
    clusters = []
    visited = [False] * len(points)
    for i in range(len(points)):
        if visited[i]:
            continue
        # Nuovo cluster
        cluster_indices = [i]
        visited[i] = True
        # Espandi cluster cercando vicini
        j = 0
        while j < len(cluster_indices):
            idx = cluster_indices[j]
            px, py = points[idx]
            # Trova punti non visitati vicini a (px, py)
            for k in range(len(points)):
                if not visited[k]:
                    qx, qy = points[k]
                    dist = math.hypot(qx - px, qy - py)
                    if dist <= dist_threshold:
                        visited[k] = True
                        cluster_indices.append(k)
            j += 1
        clusters.append(cluster_indices)
    return clusters

# Matcher BF per descrittori SIFT
bf = cv2.BFMatcher(cv2.NORM_L2)
for scene_file in scene_files:
    img_scene = cv2.imread(scenes_dir + scene_file)
    if img_scene is None:
        raise FileNotFoundError(f"Scene image {scene_file} not found")
    kp_scene, des_scene = sift.detectAndCompute(img_scene, None)
    detected_products = {}

    for mid, model_info in models_data.items():
        kp_model = model_info["kp"]
        des_model = model_info["des"]
        bary_model = model_info["barycenter"]
        h_model, w_model = model_info["size"]
        if des_model is None or des_scene is None:
            continue

        # Match SIFT (con ratio test)
        matches = bf.knnMatch(des_model, des_scene, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
        if len(good_matches) < 4:
            continue  # troppo pochi match per questo prodotto

        # Per ogni match valido, calcola la posizione prevista del baricentro nella scena
        predicted_centers = []  # lista di tuple (cx, cy) votate
        match_idx_list = []     # lista parallela di indici dei match (per riferire in seguito)
        for m in good_matches:
            # Coordinate della feature del modello e della scena
            x_m, y_m = kp_model[m.queryIdx].pt
            x_s, y_s = kp_scene[m.trainIdx].pt
            # Calcola vettore dal keypoint del modello al baricentro del modello
            bx_m, by_m = bary_model
            vec_model = (bx_m - x_m, by_m - y_m)
            # Calcola fattore di scala tra la feature del modello e della scena
            # (utilizziamo la scala dei keypoint SIFT, attributo size)
            size_model = kp_model[m.queryIdx].size
            size_scene = kp_scene[m.trainIdx].size
            scale_ratio = 1.0
            if size_model > 0:
                scale_ratio = size_scene / size_model
            # Applica il vettore scalato alla posizione della feature nella scena per ottenere voto del centro
            cx_pred = x_s + vec_model[0] * scale_ratio
            cy_pred = y_s + vec_model[1] * scale_ratio
            predicted_centers.append((cx_pred, cy_pred))
            match_idx_list.append(m)

        # Clusterizza i voti dei centri per trovare istanze distinte
        clusters = cluster_points(predicted_centers, dist_threshold=20.0)  # soglia in pixel
        # Analizza ogni cluster trovato
        for cluster_indices in clusters:
            if len(cluster_indices) < 4:
                # ignoriamo cluster deboli con pochi voti
                continue
            # Estrai i match corrispondenti a questo cluster
            cluster_matches = [match_idx_list[i] for i in cluster_indices]
            # Calcola omografia usando solo questi match (per affinare bounding box)
            src_pts = np.float32([kp_model[m.queryIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_scene[m.trainIdx].pt for m in cluster_matches]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if M is not None:
                # Trasforma i corner del modello con l'omografia per ottenere il bounding box
                corners_model = np.float32([[0, 0], [w_model, 0], [w_model, h_model], [0, h_model]]).reshape(-1, 1, 2)
                corners_scene = cv2.perspectiveTransform(corners_model, M)
                x_coords = corners_scene[:, 0, 0]
                y_coords = corners_scene[:, 0, 1]
                x_min, x_max = x_coords.min(), x_coords.max()
                y_min, y_max = y_coords.min(), y_coords.max()
                bbox_w = int(round(x_max - x_min))
                bbox_h = int(round(y_max - y_min))
                center_x = int(round(x_min + bbox_w / 2))
                center_y = int(round(y_min + bbox_h / 2))
            else:
                # Se omografia fallisce, approssima centro come media dei voti e dimensioni scalate medie
                cx_vals = [predicted_centers[i][0] for i in cluster_indices]
                cy_vals = [predicted_centers[i][1] for i in cluster_indices]
                center_x = int(round(np.mean(cx_vals)))
                center_y = int(round(np.mean(cy_vals)))
                # Stima dimensioni bounding box usando scala media dei match
                scale_avg = np.mean([ (kp_scene[m.trainIdx].size / kp_model[m.queryIdx].size) if kp_model[m.queryIdx].size>0 else 1.0 
                                       for m in cluster_matches ])
                bbox_w = int(round(w_model * scale_avg))
                bbox_h = int(round(h_model * scale_avg))
            # Salva risultato
            detected_products.setdefault(mid, []).append((center_x, center_y, bbox_w, bbox_h))

    # Stampa i risultati per l'immagine di scena corrente
    print(f"Risultati per {scene_file}:")
    if not detected_products:
        print("  Nessun prodotto riconosciuto.")
    else:
        for product_id in sorted(detected_products.keys()):
            instances = detected_products[product_id]
            count = len(instances)
            if count == 1:
                print(f"  Product {product_id} - 1 instance found:")
            else:
                print(f"  Product {product_id} - {count} instances found:")
            for idx, (cx, cy, w, h) in enumerate(instances, start=1):
                print(f"    Instance {idx} {{position: ({cx},{cy}), width: {w}px, height: {h}px}}")

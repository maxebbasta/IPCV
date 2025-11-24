import numpy as np
import cv2
from matplotlib import pyplot as plt

# --------------------------------------------------------
# Configurazione
# --------------------------------------------------------
MODELS_DIR = "Models"   # cartella dei prodotti
SCENES_DIR = "Scenes"   # cartella delle scene

# product images richieste dallo Step A
product_ids = [0, 1, 11, 19, 24, 25, 26]

# scene images richieste dallo Step A
scene_names = ["e1.png", "e2.png", "e3.png", "e4.png", "e5.png"]

# soglia minima di match buoni per stimare la omografia (come nel lab)
MIN_MATCH_COUNT = 1


# --------------------------------------------------------
# Funzione di utilità: rileva un prodotto in una scena
# (usa ESATTAMENTE la pipeline del LabSession5: SIFT + FLANN + Lowe + RANSAC)
# --------------------------------------------------------
def detect_product_in_scene(img_query, img_train):
    """
    img_query: immagine del modello (prodotto) in scala di grigi
    img_train: immagine della scena (scaffale) in scala di grigi

    Ritorna:
      - bbox_info: (cx, cy, width, height) in pixel, oppure None se non trovato
      - img_train_with_box: immagine della scena con il bounding box disegnato (o None)
      - info_matches: dizionario con info su #matches, #good, #inliers
    """

    # 1) Crea il rilevatore SIFT (come nel lab)
    sift = cv2.SIFT_create()

    # 2) Rileva i keypoint in query e train (scene)
    kp_query = sift.detect(img_query)
    kp_train = sift.detect(img_train)

    # 3) Calcola i descrittori SIFT
    kp_query, des_query = sift.compute(img_query, kp_query)
    kp_train, des_train = sift.compute(img_train, kp_train)

    # Controllo: se non ci sono descrittori, esco
    if des_query is None or des_train is None:
        return None, None, {"raw_matches": 0, "good_matches": 0, "inliers": 0}

    # 4) Inizializza FLANN (esattamente come nel notebook)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # 5) knnMatch: per ogni descrittore di query, i 2 più vicini
    matches = flann.knnMatch(des_query, des_train, k=2)

    # 6) Lowe ratio test per filtrare falsi match (soglia 0.7 come nel lab)
    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)

    raw_matches = len(matches)
    good_matches = len(good)

    # 7) Se abbastanza match buoni, stimo omografia con RANSAC
    if good_matches > MIN_MATCH_COUNT:
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_train[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Omografia robusta con RANSAC (come nel lab)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            return None, None, {"raw_matches": raw_matches,
                                "good_matches": good_matches,
                                "inliers": 0}

        matchesMask = mask.ravel().tolist()
        inliers = int(np.sum(mask))

        # 8) Proietto gli angoli dell’immagine query nella scena
        h, w = img_query.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)  # 4 punti nella scena

        # 9) Disegno il poligono (bounding box proiettato)
        img_train_color = cv2.cvtColor(img_train, cv2.COLOR_GRAY2BGR)
        img_train_color = cv2.polylines(img_train_color,
                                        [np.int32(dst)],
                                        True,
                                        (0, 255, 0), 3,
                                        cv2.LINE_AA)

        # 10) Estraggo bounding box axis-aligned per posizione, larghezza, altezza
        dst_pts = dst.reshape(-1, 2)
        xs = dst_pts[:, 0]
        ys = dst_pts[:, 1]

        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))

        width = x_max - x_min
        height = y_max - y_min
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2

        bbox_info = (cx, cy, width, height)
        info_matches = {
            "raw_matches": raw_matches,
            "good_matches": good_matches,
            "inliers": inliers
        }

        return bbox_info, img_train_color, info_matches

    else:
        # Non abbastanza match buoni
        return None, None, {"raw_matches": raw_matches,
                            "good_matches": good_matches,
                            "inliers": 0}


# --------------------------------------------------------
# Loop principale su scene e prodotti (Step A)
# --------------------------------------------------------
for scene_name in scene_names:
    print("\n==============================")
    print("Scene:", scene_name)
    print("==============================")

    # Carico la scena in scala di grigi
    scene_path = f"{SCENES_DIR}/{scene_name}"
    img_scene = cv2.imread(scene_path, 0)

    if img_scene is None:
        print("  [ERRORE] Impossibile leggere la scena:", scene_path)
        continue

    # Per visualizzare i risultati finali accumulati,
    # tengo una copia su cui disegnerò tutti i bounding box
    img_scene_result = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)

    for pid in product_ids:
        model_path = f"{MODELS_DIR}/{pid}.jpg"
        img_model = cv2.imread(model_path, 0)

        if img_model is None:
            print(f"  [ERRORE] Impossibile leggere il modello {pid}: {model_path}")
            continue

        print(f"\n  [Product {pid}]")

        bbox, img_with_box, info = detect_product_in_scene(img_model, img_scene)

        print(f"    raw matches  = {info['raw_matches']}")
        print(f"    good matches = {info['good_matches']}")
        print(f"    inliers (RANSAC) = {info['inliers']}")

        if bbox is not None:
            cx, cy, w_box, h_box = bbox
            print("    -> detection ACCEPTED")
            print(f"       center = ({cx}, {cy})")
            print(f"       width  = {w_box} px")
            print(f"       height = {h_box} px")

            # Disegno anche il box sulla scena accumulata
            # Ricostruisco i vertici del rettangolo axis-aligned
            x_min = cx - w_box // 2
            x_max = cx + w_box // 2
            y_min = cy - h_box // 2
            y_max = cy + h_box // 2

            cv2.rectangle(img_scene_result,
                          (x_min, y_min),
                          (x_max, y_max),
                          (0, 255, 0),
                          2)

            # scrivo il product id
            cv2.putText(img_scene_result,
                        f"ID {pid}",
                        (x_min, max(y_min - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA)

        else:
            print("    -> detection REJECTED (non abbastanza match buoni / inliers)")

    # Visualizzo la scena con tutti i prodotti trovati
    plt.figure(figsize=(8, 6))
    plt.title(f"Detections in {scene_name}")
    plt.imshow(cv2.cvtColor(img_scene_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
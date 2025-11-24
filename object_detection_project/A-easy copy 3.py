import numpy as np
import cv2
from matplotlib import pyplot as plt

# --------------------------------------------------------
# Configurazione
# --------------------------------------------------------
MODELS_DIR = "Models"   # cartella dei prodotti
SCENES_DIR = "Scenes"   # cartella delle scene

product_ids = [0, 1, 11, 19, 24, 25, 26]
scene_names = ["e1.png", "e2.png", "e3.png", "e4.png", "e5.png"]

# soglie per SIFT + RANSAC
MIN_MATCH_COUNT = 25      # minimo match buoni per stimare omografia
MIN_INLIERS     = 15      # minimo inliers per accettare una detection

MAX_SIDE = 1000           # ridimensionamento immagini grandi

# --------------------------------------------------------
# Istogrammi COLORE – SOLO H
# --------------------------------------------------------
def compute_histogram(img_bgr):
    """
    Istogramma colore SOLO sul canale H (HSV),
    32 bin, normalizzato.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])  # SOLO H
    cv2.normalize(hist, hist)
    return hist


def compute_scene_hist(img_scene_color, bbox):
    (cx, cy, w, h) = bbox

    x_min = cx - w // 2
    x_max = cx + w // 2
    y_min = cy - h // 2
    y_max = cy + h // 2

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_scene_color.shape[1], x_max)
    y_max = min(img_scene_color.shape[0], y_max)

    patch = img_scene_color[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return None

    return compute_histogram(patch)


def iou(bboxA, bboxB):
    (cx1, cy1, w1, h1) = bboxA
    (cx2, cy2, w2, h2) = bboxB

    x1A, y1A = cx1 - w1//2, cy1 - h1//2
    x2A, y2A = cx1 + w1//2, cy1 + h1//2

    x1B, y1B = cx2 - w2//2, cy2 - h2//2
    x2B, y2B = cx2 + w2//2, cy2 + h2//2

    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = w1*h1
    areaB = w2*h2
    union = areaA + areaB - interArea
    if union == 0:
        return 0.0

    return interArea / union


# --------------------------------------------------------
# Pre-calcolo istogrammi modelli
# --------------------------------------------------------
model_histograms = {}
for pid in product_ids:
    path = f"{MODELS_DIR}/{pid}.jpg"
    img_model_color = cv2.imread(path)
    if img_model_color is not None:
        model_histograms[pid] = compute_histogram(img_model_color)
    else:
        print(f"[ATTENZIONE] Impossibile caricare modello a colori per istogramma: {path}")


# --------------------------------------------------------
# SIFT detection
# --------------------------------------------------------
def detect_product_in_scene(img_query, img_train):

    sift = cv2.SIFT_create()

    kp_query = sift.detect(img_query)
    kp_train = sift.detect(img_train)

    kp_query, des_query = sift.compute(img_query, kp_query)
    kp_train, des_train = sift.compute(img_train, kp_train)

    if des_query is None or des_train is None:
        return None, None, {"raw_matches":0, "good_matches":0, "inliers":0}

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des_query, des_train, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.4 * n.distance:
            good.append(m)

    raw_matches = len(matches)
    good_matches = len(good)

    # richiedo almeno MIN_MATCH_COUNT match buoni
    if good_matches >= MIN_MATCH_COUNT:

        src_pts = np.float32([kp_query[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        dst_pts = np.float32([kp_train[m.trainIdx].pt for m in good]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            return None, None, {
                "raw_matches":raw_matches,
                "good_matches":good_matches,
                "inliers":0
            }

        inliers = int(np.sum(mask))

        # se ho troppo pochi inliers, omografia non affidabile
        if inliers < MIN_INLIERS:
            return None, None, {
                "raw_matches":raw_matches,
                "good_matches":good_matches,
                "inliers":inliers
            }

        h, w = img_query.shape
        pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)

        dst = cv2.perspectiveTransform(pts, M)

        dst_pts = dst.reshape(-1,2)
        xs = dst_pts[:,0]
        ys = dst_pts[:,1]

        x_min = int(np.min(xs))
        x_max = int(np.max(xs))
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))

        width  = x_max - x_min
        height = y_max - y_min
        cx = (x_min + x_max)//2
        cy = (y_min + y_max)//2

        bbox = (cx, cy, width, height)

        return bbox, None, {
            "raw_matches":raw_matches,
            "good_matches":good_matches,
            "inliers":inliers
        }

    # pochi match → niente omografia
    return None, None, {
        "raw_matches":raw_matches,
        "good_matches":good_matches,
        "inliers":0
    }


# --------------------------------------------------------
# LOOP principale
# --------------------------------------------------------
for scene_name in scene_names:

    print("\n==============================")
    print("Scene:", scene_name)
    print("==============================")

    img_scene = cv2.imread(f"{SCENES_DIR}/{scene_name}", 0)
    img_scene_color = cv2.imread(f"{SCENES_DIR}/{scene_name}")

    if img_scene is None:
        print("ERRORE lettura scena.")
        continue

    # Ridimensionamento scena
    h_s, w_s = img_scene.shape
    if max(h_s, w_s) > MAX_SIDE:
        scale = MAX_SIDE / max(h_s, w_s)
        img_scene = cv2.resize(img_scene, None, fx=scale, fy=scale)
        img_scene_color = cv2.resize(img_scene_color, None, fx=scale, fy=scale)

    img_scene_result = img_scene_color.copy()

    detections = []

    # DETECTION per ogni prodotto
    for pid in product_ids:

        print(f"\n  [Product {pid}]")

        img_model = cv2.imread(f"{MODELS_DIR}/{pid}.jpg", 0)
        if img_model is None:
            print("  ERRORE modello")
            continue

        # ridimensionamento modello
        h_m, w_m = img_model.shape
        if max(h_m, w_m) > MAX_SIDE:
            scale = MAX_SIDE / max(h_m, w_m)
            img_model = cv2.resize(img_model, None, fx=scale, fy=scale)

        bbox, _, info = detect_product_in_scene(img_model, img_scene)

        print(f"    raw matches  = {info['raw_matches']}")
        print(f"    good matches = {info['good_matches']}")
        print(f"    inliers      = {info['inliers']}")

        # qui uso di nuovo MIN_INLIERS per coerenza
        if bbox is None or info["inliers"] < MIN_INLIERS:
            print("    -> REJECTED (troppi pochi inliers o bbox nulla)")
            continue

        print("    -> CANDIDATE")

        scene_hist = compute_scene_hist(img_scene_color, bbox)

        detections.append({
            "pid": pid,
            "bbox": bbox,
            "inliers": info["inliers"],
            "scene_hist": scene_hist
        })

    # --------------------------------------------------------
    # DISAMBIGUAZIONE colore
    # --------------------------------------------------------
    final_detections = []
    IOU_THRESHOLD = 0.6

    for det in detections:
        pid  = det["pid"]
        bbox = det["bbox"]
        scene_hist = det["scene_hist"]

        if pid not in model_histograms or scene_hist is None:
            det["sim"] = 0.0
        else:
            det["sim"] = float(cv2.compareHist(model_histograms[pid],
                                               scene_hist,
                                               cv2.HISTCMP_CORREL))

        keep = True
        for kept in final_detections:
            if iou(bbox, kept["bbox"]) > IOU_THRESHOLD:
                if det["sim"] > kept["sim"]:
                    final_detections.remove(kept)
                else:
                    keep = False
                break

        if keep:
            final_detections.append(det)

    # --------------------------------------------------------
    # Disegno
    # --------------------------------------------------------
    for det in final_detections:
        pid = det["pid"]
        cx, cy, w_box, h_box = det["bbox"]

        x_min = cx - w_box//2
        x_max = cx + w_box//2
        y_min = cy - h_box//2
        y_max = cy + h_box//2

        cv2.rectangle(img_scene_result,
                      (x_min, y_min),
                      (x_max, y_max),
                      (0,255,0), 5)

        cv2.putText(img_scene_result,
            f"ID {pid}",
            (cx - 15, cy),     # TESTO CENTRATO
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,255,0), 5)


        print(f"[FINAL] Product {pid} "
              f"center=({cx},{cy}) w={w_box} h={h_box} sim={det['sim']:.3f}")

    plt.figure(figsize=(8,6))
    plt.title(f"Detections in {scene_name}")
    plt.imshow(cv2.cvtColor(img_scene_result, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

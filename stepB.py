import numpy as np
import cv2
from matplotlib import pyplot as plt


MODELS_DIR = "Models"
SCENES_DIR = "Scenes"

product_ids = [0, 1, 11, 19, 24, 25, 26]
scene_names = ["m1.png", "m2.png", "m3.png", "m4.png", "m5.png"]


MIN_MATCH_COUNT    = 1
RATIO_TEST         = 0.8
MIN_VOTES          = 3
SUPPRESSION_RADIUS = 10

MAX_SIDE           = 1000
IOU_THRESHOLD      = 0.4


# Color histogram - Only channel H
def compute_histogram_H(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [32], [0, 180])
    cv2.normalize(hist, hist)
    return hist


# IoU between bbox (cx, cy, w, h)
def iou(bboxA, bboxB):
    (cx1, cy1, w1, h1) = bboxA
    (cx2, cy2, w2, h2) = bboxB

    x1A, y1A = cx1 - w1 // 2, cy1 - h1 // 2
    x2A, y2A = cx1 + w1 // 2, cy1 + h1 // 2

    x1B, y1B = cx2 - w2 // 2, cy2 - h2 // 2
    x2B, y2B = cx2 + w2 // 2, cy2 + h2 // 2

    xA = max(x1A, x1B)
    yA = max(y1A, y1B)
    xB = min(x2A, x2B)
    yB = min(y2A, y2B)

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    if interArea == 0:
        return 0.0

    areaA = w1 * h1
    areaB = w2 * h2
    union = areaA + areaB - interArea
    if union == 0:
        return 0.0

    return interArea / union


# Models building for GHT:
# SIFT + descriptors + vectors towards barycenter
def build_model_data(image_gray, sift):

    kps, des = sift.detectAndCompute(image_gray, None)

    if des is None or len(kps) == 0:
        return None


    pts = np.array([kp.pt for kp in kps], dtype=np.float32)
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])
    center = np.array([cx, cy], dtype=np.float32)

    vectors = center - pts

    return {
        "kps": kps,
        "des": des,
        "center": center,
        "vectors": vectors,
        "shape": image_gray.shape
    }


# GHT for a scene and a single model
def detect_instances_ght(model_data, img_scene_gray, sift):

    kps_scene, des_scene = sift.detectAndCompute(img_scene_gray, None)
    if des_scene is None or len(kps_scene) == 0:
        return []

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(model_data["des"], des_scene, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < RATIO_TEST * n.distance:
            good_matches.append(m)

    if len(good_matches) < MIN_MATCH_COUNT:
        return []

    h_s, w_s = img_scene_gray.shape
    acc_votes  = np.zeros((h_s, w_s), dtype=np.int32)
    acc_scales = np.zeros((h_s, w_s), dtype=np.float32)

    vectors   = model_data["vectors"]
    kps_model = model_data["kps"]

    for m in good_matches:
        idx_model = m.queryIdx
        idx_scene = m.trainIdx

        kp_m = kps_model[idx_model]
        kp_s = kps_scene[idx_scene]

        x_s, y_s = kp_s.pt

        v = vectors[idx_model]

        if kp_m.size > 1e-6:
            scale = kp_s.size / kp_m.size
        else:
            scale = 1.0

        cx_vote = x_s + scale * v[0]
        cy_vote = y_s + scale * v[1]

        ix = int(round(cx_vote))
        iy = int(round(cy_vote))

        if 0 <= ix < w_s and 0 <= iy < h_s:
            acc_votes[iy, ix]  += 1
            acc_scales[iy, ix] += scale

    detections = []

    acc_votes_copy  = acc_votes.copy()
    acc_scales_copy = acc_scales.copy()

    while True:
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(acc_votes_copy)
        if maxVal < MIN_VOTES:
            break

        peak_x, peak_y = maxLoc
        votes_here = maxVal

        if acc_votes_copy[peak_y, peak_x] > 0:
            avg_scale = acc_scales_copy[peak_y, peak_x] / float(acc_votes_copy[peak_y, peak_x])
        else:
            avg_scale = 1.0

        detections.append({
            "center": (peak_x, peak_y),
            "scale": float(avg_scale),
            "votes": int(votes_here)
        })

        yy, xx = np.ogrid[:h_s, :w_s]
        dist2  = (xx - peak_x)**2 + (yy - peak_y)**2
        mask   = dist2 <= (SUPPRESSION_RADIUS**2)
        acc_votes_copy[mask]  = 0
        acc_scales_copy[mask] = 0.0

    return detections


# MAIN: loop on scenes e products (STEP B with color + IoU)
if __name__ == "__main__":

    sift = cv2.SIFT_create()

    models_data      = {}
    model_histograms = {}

    for pid in product_ids:
        model_path = f"{MODELS_DIR}/{pid}.jpg"

        img_model_gray  = cv2.imread(model_path, 0)
        img_model_color = cv2.imread(model_path)

        if img_model_gray is None or img_model_color is None:
            print(f"[ERROR] Impossible read the model {pid}: {model_path}")
            continue

        data = build_model_data(img_model_gray, sift)
        if data is None:
            print(f"[ATTENTION] No keypoint per model {pid}")
            continue

        models_data[pid]      = data
        model_histograms[pid] = compute_histogram_H(img_model_color)

        print(f"Model {pid}: {len(data['kps'])} keypoint")

    for scene_name in scene_names:
        print("\n==============================")
        print("Scene:", scene_name)
        print("==============================")

        scene_path = f"{SCENES_DIR}/{scene_name}"
        img_scene_gray  = cv2.imread(scene_path, 0)
        img_scene_color = cv2.imread(scene_path)

        if img_scene_gray is None or img_scene_color is None:
            print(f"[ERROR] Impossible read scene: {scene_path}")
            continue

        h_s, w_s = img_scene_gray.shape
        if max(h_s, w_s) > MAX_SIDE:
            scale_scene = MAX_SIDE / max(h_s, w_s)
            img_scene_gray = cv2.resize(img_scene_gray, None,
                                        fx=scale_scene, fy=scale_scene,
                                        interpolation=cv2.INTER_AREA)
            img_scene_color = cv2.resize(img_scene_color, None,
                                         fx=scale_scene, fy=scale_scene,
                                         interpolation=cv2.INTER_AREA)
            h_s, w_s = img_scene_gray.shape

        img_result = img_scene_color.copy()

        detections = []

        for pid, model_data in models_data.items():
            print(f"\n[Product {pid}]")

            inst_list = detect_instances_ght(model_data, img_scene_gray, sift)

            if len(inst_list) == 0:
                print("  No istance found.")
                continue

            h_m, w_m = model_data["shape"]
            print(f"  Found instances (GHT): {len(inst_list)}")

            for det in inst_list:
                cx, cy = det["center"]
                s      = det["scale"]
                votes  = det["votes"]

                w_box = int(round(w_m * s))
                h_box = int(round(h_m * s))

                x_min = int(cx - w_box / 2)
                x_max = int(cx + w_box / 2)
                y_min = int(cy - h_box / 2)
                y_max = int(cy + h_box / 2)

                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w_s - 1, x_max)
                y_max = min(h_s - 1, y_max)

                patch = img_scene_color[y_min:y_max, x_min:x_max]
                if patch.size == 0:
                    continue

                if pid in model_histograms:
                    scene_hist = compute_histogram_H(patch)
                    sim = cv2.compareHist(model_histograms[pid],
                                          scene_hist,
                                          cv2.HISTCMP_CORREL)
                else:
                    sim = 0.0

                print(f"    CANDIDATE center=({cx},{cy}) "
                      f"w={w_box} h={h_box} votes={votes} sim={sim:.3f}")

                detections.append({
                    "pid": pid,
                    "bbox": (cx, cy, w_box, h_box),
                    "votes": votes,
                    "scale": s,
                    "sim": float(sim)
                })


        # Disambiguation
        final_detections = []
        VOTE_MARGIN = 1

        for det in detections:
            pid   = det["pid"]
            bbox  = det["bbox"]
            sim   = det["sim"]
            votes = det["votes"]

            keep = True
            for kept in final_detections:
                if iou(bbox, kept["bbox"]) > IOU_THRESHOLD:
                    if votes > kept["votes"] + VOTE_MARGIN:
                        final_detections.remove(kept)

                    elif kept["votes"] > votes + VOTE_MARGIN:
                        keep = False

                    else:
                        if sim > kept["sim"]:
                            final_detections.remove(kept)
                        else:
                            keep = False

                    break

            if keep:
                final_detections.append(det)


        # Detection Draw
        for det in final_detections:
            pid          = det["pid"]
            (cx, cy, w_box, h_box) = det["bbox"]
            votes        = det["votes"]
            s            = det["scale"]
            sim          = det["sim"]

            x_min = cx - w_box // 2
            x_max = cx + w_box // 2
            y_min = cy - h_box // 2
            y_max = cy + h_box // 2

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w_s - 1, x_max)
            y_max = min(h_s - 1, y_max)

            cv2.rectangle(img_result,
                          (x_min, y_min), (x_max, y_max),
                          (0, 255, 0), 2)

            text = f"ID {pid}"
            (text_w, text_h), _ = cv2.getTextSize(text,
                                                  cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.5, 1)
            cv2.putText(img_result,
                        text,
                        (cx - text_w // 2, cy + text_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0), 1)

            print(f"[FINAL] Product {pid} center=({cx},{cy}) "
                  f"w={w_box} h={h_box} votes={votes} sim={sim:.3f} scale={s:.2f}")

        # visualize result for the actual scene
        plt.figure(figsize=(8, 6))
        plt.title(f"STEP B - Detections in {scene_name}")
        plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()


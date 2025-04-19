import cv2
import numpy as np

# Parametri
scene_files = ["h1.jpg", "h2.jpg", "h3.jpg", "h4.jpg", "h5.jpg"]  # immagini intero scaffale
model_cfg = "yolo_custom.cfg"     # path al file di configurazione YOLO addestrato sui prodotti (da 0.jpg a 23.jpg)
model_weights = "yolo_custom.weights"  # path ai pesi del modello YOLO addestrato

# Parametri di soglia per YOLO
conf_threshold = 0.5  # soglia di confidenza minima per accettare una detection
nms_threshold = 0.4   # soglia IoU per Non-Maxima Suppression

# Carica la rete neurale YOLO pre-addestrata
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # usa CPU; se GPU disponibile, cambiare preferenze

# Ottieni i nomi dei layer di output della rete YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

for scene_file in scene_files:
    img = cv2.imread(scene_file)
    if img is None:
        raise FileNotFoundError(f"Scene image {scene_file} not found")
    height, width = img.shape[:2]

    # Prepara il blob di input (ridimensiona l'immagine a 416x416, normalizza i pixel)
    blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    # Esegui la rete (forward pass)
    outputs = net.forward(output_layers)

    # Liste per le detection trovate
    boxes = []
    confidences = []
    class_ids = []

    # Elabora le uscite YOLO
    for output in outputs:
        for detection in output:
            # Le prime 4 componenti sono bbox (center_x, center_y, width, height) normalizzate [0,1], 
            # dalla quinta in poi ci sono le confidence per ogni classe
            scores = detection[5:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])
            if confidence > conf_threshold:
                # Calcola coordinate e dimensioni in termini di pixel dell'immagine originale
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Calcola coordinate dell'angolo in alto a sinistra
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

    # Applica Non-Maxima Suppression per rimuovere duplicati (per classe)
    final_indices = []
    # Raggruppa box per classe per applicare NMS separatamente su ciascuna classe
    unique_classes = set(class_ids)
    for cls in unique_classes:
        # Indici delle detection di questa classe
        idxs_cls = [i for i, cid in enumerate(class_ids) if cid == cls]
        boxes_cls = [boxes[i] for i in idxs_cls]
        confs_cls = [confidences[i] for i in idxs_cls]
        # Applica NMS per la classe corrente
        idxs = cv2.dnn.NMSBoxes(boxes_cls, confs_cls, conf_threshold, nms_threshold)
        for idx in idxs:
            final_indices.append(idxs_cls[idx[0]])

    # Organizza i risultati per prodotto
    detected_products = {}
    for i in final_indices:
        cls_id = class_ids[i]
        x, y, w, h = boxes[i]
        cx = x + w // 2
        cy = y + h // 2
        detected_products.setdefault(cls_id, []).append((cx, cy, w, h))

    # Stampa i risultati per l'immagine corrente
    print(f"Risultati per {scene_file}:")
    if not detected_products:
        print("  Nessun prodotto riconosciuto.")
    else:
        for product_id in sorted(detected_products.keys()):
            inst_list = detected_products[product_id]
            count = len(inst_list)
            if count == 1:
                print(f"  Product {product_id} - 1 instance found:")
            else:
                print(f"  Product {product_id} - {count} instances found:")
            for idx, (cx, cy, w, h) in enumerate(inst_list, start=1):
                print(f"    Instance {idx} {{position: ({cx},{cy}), width: {w}px, height: {h}px}}")

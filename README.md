# 🛒 Product Recognition on Store Shelves

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Ultralytics-orange.svg)](https://github.com/ultralytics/yolov5)

> **Authors:** Massimo Modesti, Federico Tampieri

## 📖 Overview

This repository contains a complete computer vision pipeline designed to detect and recognize product packages (e.g., cereal boxes) on crowded supermarket shelves. The project was developed in three progressive steps, evolving from classical feature-matching algorithms to advanced Deep Learning techniques using synthetic data generation.

The objective is to accurately identify the presence, count, and precise location (bounding boxes) of target product models within challenging shelf scenes featuring varying lighting, occlusions, and visual clutter.

---

## 🚀 Project Architecture

### Step A: Multiple Product Detection
**Goal:** Determine if a specific product is present in a scene and estimate its bounding box.
*   **Feature Extraction & Matching:** Uses **SIFT** (Scale-Invariant Feature Transform) to detect keypoints and a FLANN-based k-NN matcher to find correspondences between the reference model and the shelf scene.
*   **Geometric Consistency:** Applies **RANSAC** to compute a homography matrix, filtering out outlier matches and projecting the model's corners into the scene to form a bounding box.
*   **Color Disambiguation:** SIFT operates in grayscale, which causes false positives for products with identical packaging but different colors (e.g., Kellogg's Choco Krave Milk vs. Dark Chocolate). We extract the **Hue (H) channel histogram** from the HSV color space and use correlation-based similarity (`cv2.compareHist`) to resolve color ambiguities.
*   **Non-Maximum Suppression:** Overlapping detections are merged using an **Intersection over Union (IoU)** threshold (0.6), prioritizing the candidate with the highest color similarity.

### Step B: Multiple Instance Detection
**Goal:** Detect *all* instances of a specific product on the same shelf.
*   **Generalized Hough Transform (GHT):** Extends Step A by implementing a voting scheme. SIFT keypoints in the scene cast "votes" for the object's barycenter based on learned spatial vectors from the model.
*   **Accumulator & Scale Estimation:** Votes are collected in a 2D accumulator array. Local maxima (peaks) indicate candidate object locations. Bounding box sizes are dynamically estimated based on the scale differences between matched keypoints.
*   **Refinement:** Uses the same Hue-based color verification and an IoU threshold (0.4) combined with a vote-margin check to suppress duplicate overlapping instances.

### Step C: Whole Shelf Challenge & Sim2Real
**Goal:** Robust detection of densely packed items using Deep Learning (**YOLOv5**) without manual labeling.
*   **Sim2Real Strategy:** Classical methods (SIFT + GHT) struggle with extreme occlusions and low-resolution images. We developed a robust **Synthetic Data Generator** to train a YOLOv5s model entirely on artificial data.
*   **Domain Randomization Pipeline:**
    1.  **Geometric Scaling:** Simulates varying camera distances.
    2.  **Photometric Augmentation:** Adjusts RGB brightness/contrast and HSV saturation.
    3.  **Synthetic Occlusion:** Randomly crops templates using the Alpha channel to force the network to learn partial features.
    4.  **Sensor Simulation:** Applies Gaussian blur and additive noise.
    5.  **Perspective Warping:** Simulates 3D poses (viewing objects from the side of the aisle).
    6.  **Background Superimposition:** Blends the augmented templates onto real supermarket background images.
*   **Automatic Annotation:** The generation script automatically computes normalized YOLO-format labels with pixel-perfect accuracy, yielding a dataset of 30,000 annotated images.
*   **Results:** The YOLOv5 model achieved an **mAP@0.5 of 0.995** on synthetic validation and demonstrated exceptional zero-shot generalization to real-world store shelf images.

---

## 📂 Repository Structure

```text
├── Models/                     # Reference images of the target products (.jpg)
├── Scenes/                     # Real test images of supermarket shelves (.png)
├── stepA.py                    # Script for Single Instance Product Detection (SIFT + RANSAC)
├── stepB.py                    # Script for Multiple Instance Detection (GHT)
├── stepC_data_generator.py     # Synthetic dataset generator with domain randomization
├── report.pdf                  # Comprehensive academic report detailing the methodology
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. The primary dependencies are `numpy`, `opencv-python`, and `matplotlib`.

```bash
pip install numpy opencv-python matplotlib
```

### Running the Classical Pipelines (Step A & B)
Ensure that the `Models/` and `Scenes/` directories are populated with the target images and shelf scenes.

```bash
# Run Multiple Product Detection
python stepA.py

# Run Multiple Instance Detection
python stepB.py
```

### Running the Synthetic Data Generator (Step C)
To generate the synthetic dataset for YOLO training:
1. Create a `dataset/backgrounds/` folder and add some empty shelf background images.
2. Run the generator script:

```bash
python stepC_data_generator.py
```
This will create `dataset/images/train/` and `dataset/labels/train/` with the synthetic scenes and corresponding YOLO `.txt` labels.

### YOLOv5 Inference
After training your YOLOv5 model on the synthetic dataset, you can run inference on real shelf images:

```bash
python detect.py     --weights runs/train/stepC_augmented/weights/best.pt     --source path/to/real_scenes/     --imgsz 640     --conf-thres 0.15     --iou-thres 0.45
```

---

## 📊 Performance & Insights

*   **Color is Critical:** In Step A, leveraging Hue histograms prevented false positives between geometrically identical but color-distinct products.
*   **GHT for Repetition:** The Generalized Hough Transform effectively allowed SIFT to detect multiple copies of the same item, overcoming standard homography limitations.
*   **Synthetic Data Works:** By heavily randomizing transformations, lighting, blur, and occlusions, we completely bridged the Sim2Real gap. The YOLO model learned object-specific features rather than overfitting to synthetic artifacts, resulting in near-perfect localization on real shelves.

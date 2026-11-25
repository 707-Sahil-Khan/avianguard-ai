# AvianGuard AI — Smart Aviary Health Monitor

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20App-FF4B4B)

**AvianGuard AI** is a multi-modal deep learning monitoring system built with **PyTorch** and **Streamlit**. It integrates computer vision, acoustic signal processing, and spatiotemporal video analysis into a unified 4-tab web dashboard to screen poultry and avian health indicators.

> **Disclaimer:** This project is a classroom and research benchmark trained on small public datasets. It is intended strictly for educational and demonstration purposes and is **not** a clinical or veterinary diagnostic tool[cite: 2].

---

## Table of Contents
- [System Architecture & Modules](#system-architecture--modules)
- [Repository Structure](#repository-structure)
- [Expected Model & Evaluation Artifacts](#expected-model--evaluation-artifacts)
- [Installation & Setup](#installation--setup)
- [Running the Application](#running-the-application)
- [Pushing Future Updates to GitHub](#pushing-future-updates-to-github)

---

## System Architecture & Modules

The web dashboard (`app/app.py`) is organized into four interactive modules[cite: 2]:

### 1. Visual Triage & Droppings Pathology Scanner (`Module 3`)
* **Task:** Multi-class image classification of poultry droppings (`JPG`, `JPEG`, `PNG`)[cite: 2].
* **Architectures:** `VGG16` and `AlexNet` (both transfer-learned and trained from scratch)[cite: 2].
* **Preprocessing:** Resized to `224 × 224` RGB and normalized with ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)[cite: 2].
* **Explainability:** Live **Grad-CAM** activation heatmaps (`pytorch-grad-cam`) overlaid on the input image, with automatic fallback to `gradcam_demo.png` if live Grad-CAM is unavailable[cite: 2].

### 2. Acoustic Cough & Distress Listener (`Module 2`)
* **Task:** Binary/multi-class respiratory vocalization screening from `.wav` farm audio clips (up to 18 seconds at `16 kHz`)[cite: 2].
* **Architectures:**
  * `VGG1D`: 1D convolutional network with batch normalization and adaptive average pooling[cite: 2].
  * `Alex1D`: 1D AlexNet-style convolutional feature extractor[cite: 2].
  * `AudioRNN`: 2-layer `LSTM` (`hidden_size=128`, `dropout=0.3`) over transposed MFCC sequences[cite: 2].
* **Preprocessing:** Splits audio into up to 6 non-overlapping 3-second segments (`48,000` samples each), extracts `40` MFCC bands (`hop_length=512`), standardizes using saved training `mu` and `sd`, and averages softmax probabilities across segments[cite: 2].
* **Visualizations:** Displays time-domain waveform, log-power Mel spectrogram, class probability distribution, and an automated **RESPIRATORY ALERT** banner when an `"unhealthy"` class is predicted[cite: 2].

### 3. Video Behavior & Activity Tracker (`Module 1`)
* **Task:** Temporal classification across 7 avian behaviors (`Feeding`, `Preening`, `Swimming`, `Walking`, `Alert`, `Flying`, `Resting`) alongside frame-difference inactivity detection (`MP4`, `MOV`, `AVI`)[cite: 2].
* **Architectures:** Frozen `VGG16` or `AlexNet` spatial feature extractors (`features` backbone) paired with temporal sequence heads (`LSTM` or vanilla `RNN`, `hidden_size=128`, `dropout=0.3`)[cite: 2].
* **Preprocessing:** Samples up to `12` sliding windows (`step=48` frames, `T=8` evenly spaced frames per window at `224 × 224`), computes global average-pooled CNN embeddings standardized by saved `mu` and `sd`, and measures mean grayscale frame-difference motion (`160 × 90`) against a configurable inactivity threshold[cite: 2].

### 4. Syllabus Model Benchmarks
* **Task:** Side-by-side comparison of held-out test performance across all three modalities[cite: 2].
* **Metrics:** Displays **Accuracy**, **Macro F1**, and interactive **Confusion Matrices** loaded from the module JSON evaluation files[cite: 2].

---

## Repository Structure

```text
avianguard-ai/
├── app/
│   └── app.py              # Main 4-tab Streamlit application & PyTorch model definitions
├── .gitignore              # Excludes raw datasets (audio/, images/, video/) & outputs/
├── requirements.txt        # Python package dependencies
└── README.md               # Project documentation

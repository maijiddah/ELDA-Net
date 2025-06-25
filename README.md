# ELDA-Net

**Edge-Lightweight Detection and Adaptation Network for Real-Time Lane Detection**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)

---

## 🚀 Overview

ELDA-Net is a lightweight, efficient lane detection system that combines deep semantic segmentation with classical vision-based enhancements. Built on a modified U-Net architecture, it is optimized for real-time performance on resource-constrained platforms like NVIDIA Jetson Nano.

---

## ✨ Key Features

- 🔍 Semantic Segmentation via Custom U-Net
- ⚡ Fast Inference on Edge Devices (25 FPS on Jetson Nano)
- 🧠 Adaptive Lane Estimation for occluded/missing lanes
- 🎥 Video overlay for real-time driving visualization
- 📊 Evaluated on TuSimple and CULane datasets

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/ELDA-Net.git
cd ELDA-Net
pip install -r requirements.txt
```

Make sure `torch`, `opencv-python`, and `PyYAML` are included in `requirements.txt`.

---

## 🧪 How to Use

### Train
```bash
python elda_net_experiment.py --mode train --dataset TuSimple
```

### Evaluate
```bash
python elda_net_experiment.py --mode eval --dataset TuSimple
```

### Inference on Video
```bash
python elda_net_experiment.py --mode infer --video demo.mp4 --output results/output.mp4
```

> Configure dataset paths and hyperparameters in `config.yaml`.

---

## 📂 Project Structure

```
ELDA-Net/
├── model/                     # U-Net architecture
├── utils/                     # Dataset, preprocessing, metrics, visualization
├── config.yaml                # Model + training configuration
├── elda_net_experiment.py     # Main entry point
├── LICENSE
├── README.md
```

---

## 📈 Results

| Dataset   | F1 Score | IoU   | FPS   |
|-----------|----------|-------|-------|
| TuSimple  | 0.932    | 0.865 | 25    |
| CULane    | 0.876    | 0.821 | 23    |

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{eldanet2025,
  title={ELDA-Net: Edge-Lightweight Detection and Adaptation Network for Real-Time Lane Detection},
  author={Abdullahi Hauwa Suleiman},
  year={2025},
  howpublished={\url{https://github.com/maijiddah/ELDA-Net}},
  note={GitHub repository}
}
```

---

## 🔒 License

This project is licensed under the [MIT License](LICENSE).

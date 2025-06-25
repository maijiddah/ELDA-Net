# 🚀 ELDA-Net v1.0.0 – Initial Public Release

## Overview
**ELDA-Net (Edge-Lightweight Detection and Adaptation Network)** is a real-time, resource-efficient lane detection framework tailored for Advanced Driver-Assistance Systems (ADAS) and autonomous driving applications. It integrates a modified U-Net architecture with adaptive lane estimation techniques, offering a balanced trade-off between accuracy and computational demand, making it suitable for deployment on edge devices such as the NVIDIA Jetson Nano.

## 🔧 Features
- ⚡ **Lightweight Architecture** based on U-Net for fast inference on embedded platforms  
- 🧠 **Adaptive Lane Estimation** for robust tracking of occluded, faded, or missing lane markers  
- 🎥 **Video Inference Support** with semantic segmentation overlay  
- 📊 **Comprehensive Evaluation** on TuSimple and CULane datasets  
- 🧪 **Modular Design** for easy experimentation and extension

## 📂 Repository Structure
```
ELDA-Net/
├── elda_net_experiment.py    # Main script for training, evaluation, and inference
├── config.yaml               # Configurable hyperparameters and dataset paths
├── README.md                 # Setup and usage guide
├── checkpoints/              # Model checkpoints (to be created after training)
├── data/                     # Dataset directories (user-supplied)
└── results/                  # Output video or predictions
```

## 🧪 How to Use

**Training:**
```bash
python elda_net_experiment.py --mode train --dataset TuSimple
```

**Evaluation:**
```bash
python elda_net_experiment.py --mode eval --dataset TuSimple
```

**Inference on Video:**
```bash
python elda_net_experiment.py --mode infer --video demo.mp4 --output results/output.mp4
```

## 🧾 Citation
If you use this project in your research, please cite:

```bibtex
@misc{eldanet2025,
  title={ELDA-Net: Edge-Lightweight Detection and Adaptation Network for Real-Time Lane Detection},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/ELDA-Net}},
  note={GitHub repository}
}
```

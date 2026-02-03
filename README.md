<div align="center">

# 🌊 Multi-Modal Depth Estimation with Semantic Segmentation

<img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch"/>
<img src="https://img.shields.io/badge/Python-3.9+-3776ab?logo=python&logoColor=white" alt="Python"/>
<img src="https://img.shields.io/badge/Dataset-NYU%20Depth%20V2-blue" alt="NYU Depth V2"/>
<img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>

*A DepthNet-style architecture for depth completion using RGB images, sparse LiDAR/depth maps, and semantic segmentation maps*

<br>

<img src="output1.png" alt="Depth Estimation Results" width="90%"/>
<br>
<img src="output2.png" alt="Depth Estimation Results 2" width="90%"/>

</div>

---

## ✨ Overview

This project implements a **deep learning pipeline** for monocular depth estimation, inspired by the DepthNet and Pix2Pix architectures. The model takes multi-modal inputs and produces dense depth maps, enabling applications in:

- 🚗 **Autonomous Driving** — Scene understanding and obstacle detection
- 🤖 **Robotics** — Navigation and spatial awareness
- 🎮 **AR/VR** — 3D scene reconstruction
- 🏠 **Indoor Mapping** — Room layout estimation

> 🧪 *This was a fun experimental project completed during the first month of my summer vacation using free GPU time on Kaggle.*

---

## 🎯 Key Features

<table>
<tr>
<td width="50%">

### 🔧 Technical Highlights

- **Multi-modal Fusion**: RGB + Sparse Depth + Semantic Segmentation
- **Encoder-Decoder Architecture**: Skip connections for detail preservation
- **Multi-scale Supervision**: Coarse-to-fine depth refinement
- **Instance Normalization**: GroupNorm for stable training
- **Dropout Regularization**: Prevents overfitting

</td>
<td width="50%">

### 📊 Performance

| Epochs | L1 Loss | Status |
|:------:|:-------:|:------:|
| 90 | ~0.120 | 🟡 Training |
| 150 | ~0.060 | 🟠 Improving |
| 250 | ~0.025 | 🟢 Good |
| 500 | ~0.008 | ✅ Converged |

</td>
</tr>
</table>

---

## 🏗️ Architecture

<div align="center">
<img src="unet_graph.png" alt="Model Architecture" width="60%"/>

*DepthNet Encoder-Decoder with Skip Connections*
</div>

### Network Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         INPUT (6 channels)                       │
│              [ RGB (3) + Sparse Depth (1) + Semantic (2) ]       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                           ENCODER                                │
│   Conv1 (32) → Conv2 (64) → Conv3 (128) → Conv4 (256) → ...     │
│              Strided convolutions + GroupNorm + ReLU             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                           DECODER                                │
│   Up5 (256) → Up4 (128) → Up3 (64) → Up2 (32) → Up1 (32)        │
│                Bilinear Upsample + Skip Connections              │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MULTI-SCALE PREDICTIONS                       │
│          Depth maps at 5 resolutions (64×64 to 256×256)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📂 Dataset

<div align="center">

### NYU Depth V2 Dataset

| Component | Description | Shape |
|:---------:|:-----------:|:-----:|
| 🖼️ RGB Images | Indoor scene photographs | 640 × 480 × 3 |
| 📏 Depth Maps | Ground truth depth | 640 × 480 |
| 🏷️ Semantic Labels | Per-pixel class annotations | 640 × 480 |
| 📦 Instance Maps | Object instance segmentation | 640 × 480 |

</div>

**Dataset Link:** [cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib pillow tqdm h5py
```

### Training

```python
# Clone the repository
git clone https://github.com/YourUsername/Depth-Estimation-with-Semantic-Segmentation.git
cd Depth-Estimation-with-Semantic-Segmentation

# Run the Jupyter notebook
jupyter notebook Model.ipynb
```

### Inference

```python
import torch
from model import DepthNet

# Load pre-trained model
model = DepthNet(batch_norm=True, dropout=0.2)
model.load_state_dict(torch.load("depthnet_final.pth"))
model.eval()

# Prepare input (6 channels: RGB + Sparse Depth + Semantic)
rgb = ...          # [B, 3, H, W]
sparse_depth = ... # [B, 1, H, W]
semantic = ...     # [B, 2, H, W]

input_tensor = torch.cat([rgb, sparse_depth, semantic], dim=1)

with torch.no_grad():
    predictions = model(input_tensor)
    final_depth = predictions[0]  # Finest resolution
```

---

## 📁 Project Structure

```
📦 Depth-Estimation-with-Semantic-Segmentation
├── 📓 Model.ipynb          # Main training notebook
├── 📊 depthnet_final.pth   # Pre-trained model weights
├── 🖼️ output1.png          # Sample prediction 1
├── 🖼️ output2.png          # Sample prediction 2
├── 📐 unet_graph.png       # Architecture visualization
├── 📋 requirements.txt     # Dependencies
└── 📖 README.md            # This file
```

---

## 🎓 Training Details

### Loss Function

```python
def depth_metric_reconstruction_loss(pred, target, normalize=False):
    """
    Multi-scale L1 loss with optional relative normalization.
    Supervision at multiple resolutions enables coarse-to-fine learning.
    """
    # Weighted sum of losses at different scales
    # Higher weight for finer resolutions
    pass
```

### Hyperparameters

| Parameter | Value |
|:---------:|:-----:|
| Image Size | 256 × 256 |
| Batch Size | 4 |
| Learning Rate | 2e-4 |
| Optimizer | Adam |
| Epochs | 500 |
| Dropout | 0.2 |

---

## 📚 References & Inspiration

<table>
<tr>
<td align="center" width="25%">
<img src="https://img.icons8.com/color/96/000000/document.png" width="48"/>
<br><b>DepthNet</b>
<br><sub>Wofk et al., ICCV 2019</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/color/96/000000/image.png" width="48"/>
<br><b>Pix2Pix</b>
<br><sub>Image-to-Image Translation</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/color/96/000000/database.png" width="48"/>
<br><b>NYU Depth V2</b>
<br><sub>Indoor Scene Dataset</sub>
</td>
<td align="center" width="25%">
<img src="https://img.icons8.com/color/96/000000/gpu.png" width="48"/>
<br><b>Kaggle</b>
<br><sub>Free GPU Compute</sub>
</td>
</tr>
</table>

---

## 🙏 Acknowledgements

<div align="center">

| | |
|:--:|:--:|
| 🎮 **Kaggle** | For providing free GPU time and a smooth training experience |
| 🏫 **NYU** | For the excellent NYU Depth V2 dataset |
| 📘 **Research Community** | For foundational work in depth estimation |

</div>

> *This project was part of my personal learning journey during summer vacation, helping me gain hands-on experience with multi-modal deep learning pipelines and loss functions for dense prediction tasks.*

---

<div align="center">

### 💡 Future Improvements

| Enhancement | Status |
|:-----------:|:------:|
| Add confidence maps | 🔜 Planned |
| Improve edge sharpness | 🔜 Planned |
| Test on outdoor scenes | 🔜 Planned |
| Add real-time inference | 🔜 Planned |

---

<br>

**Made with ❤️ and PyTorch**

<br>

<img src="https://img.shields.io/badge/⭐_Star_this_repo-if_you_found_it_helpful!-yellow?style=for-the-badge" alt="Star this repo"/>

<br>

*Feel free to fork, experiment, and improve!*

</div>

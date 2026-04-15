 📖 Introduction
TSRNet addresses a fundamental limitation in airborne LiDAR processing: existing deep learning methods optimize only for point-level binary classification and do not explicitly supervise terrain surface reconstruction quality. This means that gains in segmentation accuracy (mIoU) do not translate proportionally to improvements in DTM elevation accuracy — the geoscientific output of primary interest.

TSRNet introduces three complementary innovations to bridge this gap:

- **TCAS (Terrain Complexity-Aware Sampling):** Replaces uniform Farthest Point Sampling (FPS) with a curvature-slope-sparsity composite score that concentrates encoder sampling in geomorphologically complex zones (ridge crests, terrain breaklines, abrupt slope transitions), directly addressing the under-representation of critical terrain discontinuities.

- **BTMamba (Bidirectional Terrain Mamba):** Serializes point features along PCA-derived principal slope and contour directions before applying bidirectional selective state space (SSM) scanning, encoding the anisotropic directional continuity inherent in natural terrain at linear computational cost. Unlike Morton-code or random orderings, these sequences align with terrain-intrinsic geometry.

- **DGER (DTM-Guided Elevation Regression):** Extends the decoder with a parallel elevation regression branch jointly supervised by point-level smooth-L1 loss, a Terrain Surface Consistency Loss (TSCL) that enforces local pairwise elevation relationships among neighboring ground points, and a DTM Rasterization Loss that directly penalizes raster-level surface error.

**Performance Highlights:**

| Dataset | Benchmark | OA | mIoU |
|---|---|---|---|
| OpenGF | Test I (Mixed Terrain) | 96.35% | 92.80% |
| OpenGF | Test II (Urban) | 94.94% | 90.37% |
| OpenGF | Test III (Sparse Mountain) | 97.98% | 93.31% |
| ISPRS | Cross-dataset (15 samples) | — | 81.57% (avg) |

 💻 System Requirements

This model requires a robust hardware environment due to the multi-task joint training of segmentation and elevation regression.

- **OS:** Linux (Ubuntu 20.04 / 22.04 recommended)
- **GPU:** NVIDIA GPU with Compute Capability ≥ 8.0 (Ampere architecture or newer)
- **VRAM:**
  - Training: ≥ 24 GB (e.g., RTX 4090/5090, A100/A800)
  - Inference: ≥ 20 GB
- **CUDA:** 13.0 (strictly required for the Mamba installation steps below)

 🛠️ Installation

To ensure reproducibility, please follow these steps strictly.

**1. Clone the repository**
```bash
git clone https://github.com/MaoZhen-Li/TSRNet.git
cd TSRNet
```

**2. Create environment**
```bash
conda create -n tsrnet python=3.12
conda activate tsrnet
```

**3. Install PyTorch (CUDA 13.0)**
```bash
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

**4. Install Mamba-SSM and core dependencies**

This step requires `nvcc` (CUDA compiler) to be available in your path.

```bash
# Install core libraries
pip install -r requirements.txt

# Install Mamba components
pip install causal_conv1d-1.6.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
pip install mamba_ssm-2.3.1+cu13torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

**5. Install PointNet++ operations**
```bash
pip install pointnet2_ops
```

⚡ Usage
## Pre-trained Weights

The pre-trained model checkpoint is available on Hugging Face: https://huggingface.co/Maozhen-Li/TSRNet/blob/main/best_model.pth
### Training

**Step 1: Precompute features (recommended for faster training)**
```bash
python dataloader.py --precompute --data_root data
```

**Step 2: Launch training**
```bash
python train.py \
    --batch_size 15 \
    --epoch 60 \
    --learning_rate 0.001 \
    --log_dir TSRNet \
    --elevation_noise_aug \
    --elevation_noise_std 0.05 \
    --lambda1 0.5 \
    --lambda2 0.3 \
    --lambda4 0.2 \
    --mamba_d_state 16
    --mamba_d_conv 4
    --mamba_expand 2
```

The training script integrates a learning rate warmup (5 epochs) followed by CosineAnnealingWarmRestarts, and jointly monitors mIoU and DTM-RMSE throughout training.

### Inference / Testing

python test.py 

📂 Data Preparation

We use two publicly available ALS point cloud benchmarks.

### OpenGF Dataset

The primary training and evaluation benchmark, comprising over 500 million annotated points across nine terrain types from four countries, covering more than 47 km².

- Request access: https://github.com/Nathan-UW/OpenGF
- Qin, N., et al. "OpenGF: An ultra-large-scale ground filtering dataset built upon open ALS point clouds around the world." CVPRW 2021.

### ISPRS ALS Filter Dataset

Used for cross-dataset generalization evaluation (15 samples).

- Request access: https://www.itc.nl/isprs/wgIII-3/filtertest/downloadsites
- Sithole, G., and Vosselman, G. "Experimental comparison of filter algorithms for bare-Earth extraction from airborne laser scanning point clouds." ISPRS J. Photogramm. Remote Sens. 59(1–2), 2004.

### Directory Structure

```
data/
├── OpenGF_train/
│   ├── S1/          
│   ├── S2/           
│   ├── S3/          
│   ├── S4/          
│   ├── S5/         
│   ├── S6/
│   ├── S7/
│   ├── S8/
│   └── S9/          
├── OpenGF_test/
│   ├── TestI/       
│   ├── TestII/       
│   └── TestIII/     
└── ISPRS/
    ├── Samp11.npy
    ├── Samp12.npy
    └── ...           
```

Each `.npy` file should contain an `(N, 4)` array: `[X, Y, Z, Label]`, where `Label` is `0` for ground and `1` for non-ground.

### Point Feature Format

TSRNet uses 11-dimensional input features per point (precomputed and cached):

| Dim | Feature | Description |
|---|---|---|
| 0–2 | Centered XYZ | Block-centered Cartesian coordinates |
| 3–5 | Normalized XYZ | Global scene-normalized coordinates |
| 6 | Local density | Adaptive-radius neighborhood count, normalized |
| 7–9 | Surface normal | Weighted PCA normal |
| 10 | Curvature | Eigenvalue ratio from local covariance matrix |

---

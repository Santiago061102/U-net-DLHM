# U-Net for Digital Lensless Holographic Microscopy (DLHM)

Deep learning-based twin-image removal for digital lensless holographic microscopy. A U-Net is trained to recover clean phase or amplitude maps from holographic reconstructions affected by the twin-image artifact.

---

## Overview

In in-line digital holography, the twin image is an unwanted artifact that overlaps with the real reconstruction and degrades image quality. This project trains a U-Net to suppress it, learning a direct mapping from the noisy reconstruction to the clean phase or amplitude image.

Three model versions are included:

| Version | Architecture |
|---|---|
| v0 | U-Net (baseline) |
| v1 | Attention-Based U-Net |
| v2 | Optimized Attention-Based U-Net |

---

## Repository structure

```
.
├── train.py              ← training script
├── inference_v0.py       ← inference with v0 model
├── inference_v1.py       ← inference with v1 model
├── inference_v2.py       ← inference with v2 model
├── v1Model.py            ← U-Net architecture
├── v2Model.py            ← Attention-Based U-Net architecture
├── v3Model.py            ← Optimized Attention-Based U-Net architecture
├── utils.py              ← dataset, transforms, logger, helpers
└── losses2.py            ← LPIPS and hybrid loss functions
```

---

## Requirements

```bash
pip install torch torchvision pillow numpy matplotlib scikit-image opencv-python progress torchmetrics
```

Python ≥ 3.10 and PyTorch ≥ 2.0 recommended. GPU optional — scripts fall back to CPU automatically.

---

## Dataset structure

```
dataset/
├── src/          ← input holograms (twin-image reconstructions)
└── tar_ph/       ← ground-truth phase maps
```

Images are expected as grayscale `.png` files with matching filenames in both folders.

---

## Training

Edit the `CONFIG` section at the top of `train.py`, then run:

```bash
python train.py
```

Key parameters:

| Parameter | Default | Description |
|---|---|---|
| `DATA_PATH` | — | Path to the dataset folder |
| `NUM_IMAGES` | 28000 | Number of image pairs to load |
| `EPOCHS` | 700 | Maximum training epochs |
| `BATCH_SIZE` | 16 | Images per batch |
| `LR` | 1e-4 | Initial learning rate |
| `FILTERS` | 64 | U-Net base filter count |

Training uses an 80/20 train/validation split, `ReduceLROnPlateau` scheduling, and early stopping (patience = 20 epochs). The best model by validation loss is saved automatically.

---

## Inference

Edit the `CONFIG` section at the top of the corresponding script, then run:

```bash
python inference_v0.py   # v0 — plain U-Net
python inference_v1.py   # v1 — Attention U-Net
python inference_v2.py   # v2 — Optimized Attention U-Net
```

Each script saves a two-panel figure (`With twin image` | `Inference`) to the `output/` folder. If a ground-truth path is provided, PSNR, SSIM and MSE are printed to the console.

---

## Results

Metrics are computed on images normalized to [0, 1]:

| Metric | Description |
|---|---|
| PSNR | Peak signal-to-noise ratio (dB) — higher is better |
| SSIM | Structural similarity index — higher is better |
| MSE | Mean squared error — lower is better |

---

## License

MIT — see `LICENSE` for details.

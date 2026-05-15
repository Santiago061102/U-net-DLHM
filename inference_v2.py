"""
inference_v2.py — v2: Optimized Attention-Based UNet

Edit the paths in the CONFIG section below, then run:
    python inference_v2.py
"""

import cv2, torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_sk
from skimage.metrics import structural_similarity as ssim_sk
from utils import *
from pathlib import Path

from v2Model import UNet  # v2: Optimized Attention-Based UNet

# ---------------------------------------------------------------------------
# CONFIG — edit these paths before running
# ---------------------------------------------------------------------------

MODEL = r".\v.2 Optimized Phase Attention-Based UNet .pt"
SRC   = r".\Phase.png"
GT    = r".\tar_ph\00000_benchmark1.png"  # set to None to skip metrics

MODE  = "phase"   # "phase" or "amplitude"
SIZE  = 256
OUT   = "output"

# ---------------------------------------------------------------------------


def load_image(path, size):
    tfm = Compose([Resize((size, size)), ToTensor()])
    return tfm(Image.open(path).convert("L")).unsqueeze(0)


def to_numpy(tensor):
    img = (tensor[0, 0].cpu().numpy() + 1.0) / 2.0
    return cv2.rotate(cv2.flip(img, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)


def compute_metrics(pred, gt):
    psnr = psnr_sk(gt, pred, data_range=1.0)
    ssim = ssim_sk(gt, pred, data_range=1.0)
    mse  = float(np.mean((gt - pred) ** 2))
    print(f"  PSNR : {psnr:.2f} dB")
    print(f"  SSIM : {ssim:.4f}")
    print(f"  MSE  : {mse:.6f}")


def save_figure(images, labels, title, out_path):
    n = len(images)
    fig = plt.figure(figsize=(3 * n, 3))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.04)
    gs = gridspec.GridSpec(1, n, figure=fig, wspace=0.05)

    for col, (img, lbl) in enumerate(zip(images, labels)):
        ax = fig.add_subplot(gs[0, col])
        im = ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.annotate(lbl, xy=(0.5, -0.06), xycoords="axes fraction",
                    ha="center", fontsize=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        if col == n - 1:
            fig.colorbar(im, cax=cax)
        else:
            cax.axis("off")

    Path(out_path).parent.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.show()


# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("Model: v2 — Optimized Attention-Based UNet")

model = UNet(filters=64, attn=True).to(device)
model.load_state_dict(torch.load(MODEL, map_location=device, weights_only=True))
model.eval()

src_t = load_image(SRC, SIZE).to(device)
with torch.no_grad():
    pred_t = model(src_t)

src_np  = to_numpy(src_t)
pred_np = to_numpy(pred_t)

if MODE == "phase":
    pred_np = 1 - pred_np

if GT is not None:
    gt_np = to_numpy(load_image(GT, SIZE).to(device))
    print("\nMetrics:")
    compute_metrics(pred_np, gt_np)

images = [src_np, pred_np]
labels = ["With twin image", "Inference — v2 Optimized Attention UNet"]

title = "Phase-only" if MODE == "phase" else "Amplitude-only"
save_figure(images, labels, title, f"{OUT}/result_v2.png")

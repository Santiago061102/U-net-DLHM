"""
train.py — U-Net training for holographic reconstruction

Edit the CONFIG section below, then run:
    python train.py
"""

import torch, os, glob, json, time
from torch.utils.data import DataLoader
from progress.bar import IncrementalBar

from v2Model import UNet
from utils import *
from losses import LPIPSLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau


# ---------------------------------------------------------------------------
# CONFIG — edit before running
# ---------------------------------------------------------------------------

DATA_PATH  = '.\bigDataset'
NUM_IMAGES = 28000

EPOCHS     = 700
BATCH_SIZE = 16
LR         = 0.0001
IM_SIZE    = 256
FILTERS    = 64

LOG_NAME   = "v.2 Optimized UNet-Attention model"   # name prefix for saved weights and metrics

# ---------------------------------------------------------------------------


# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

transforms = Compose([Resize((IM_SIZE, IM_SIZE)), ToTensor(), Rotate()])

# Model
model = UNet(filters=FILTERS, attn=True).to(device)
model.apply(initialize_weights)

# Optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999))
criterion = LPIPSLoss().to(device)

# Dataset — 80% train / 20% validation split
data       = Data(root=DATA_PATH, transform=transforms, numImages=NUM_IMAGES)
train_size = int(0.8 * len(data))
val_size   = len(data) - train_size
train_set, val_set = torch.utils.data.random_split(data, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False)

# Scheduler and early stopping
scheduler     = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
early_stop    = EarlyStopping(patience=20, min_delta=2e-4)
best_val_loss = float('inf')

# Logger
args_info = {
    "epochs": EPOCHS, "batch_size": BATCH_SIZE, "lr": LR,
    "im_size": IM_SIZE, "filters": FILTERS, "num_images": NUM_IMAGES,
    "loss": "LPIPS"
}
logger = Logger(filename=LOG_NAME)
logger.historical(args_info)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

print("Start of training!")
for epoch in range(EPOCHS):
    train_loss = 0.0
    val_loss   = 0.0
    start      = time.time()

    bar = IncrementalBar(f'[Epoch {epoch+1}/{EPOCHS}]', max=len(train_loader))

    # -- Train --
    model.train()
    for src, tar in train_loader:
        src, tar = src.to(device), tar.to(device)

        pred = model(src)
        loss = criterion(pred, tar)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        bar.next()

    # -- Validate --
    model.eval()
    with torch.no_grad():
        for src, tar in val_loader:
            src, tar = src.to(device), tar.to(device)
            pred     = model(src)
            val_loss += criterion(pred, tar).item()
            del pred, src, tar

    if device == 'cuda':
        torch.cuda.empty_cache()

    bar.finish()

    train_loss /= len(train_loader)
    val_loss   /= len(val_loader)

    scheduler.step(val_loss)
    early_stop(val_loss)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        logger.save_weights(model.state_dict(), 'unet_best')
        print(f"  Best model saved (val_loss: {val_loss:.4f})")

    # Log metrics
    logger.add_scalar('train_loss', train_loss, epoch + 1)
    logger.add_scalar('val_loss',   val_loss,   epoch + 1)
    logger.save_weights(model.state_dict(), 'unet')

    elapsed = time.time() - start
    print(f"[Epoch {epoch+1}/{EPOCHS}]  train: {train_loss:.4f}  val: {val_loss:.4f}  lr: {optimizer.param_groups[0]['lr']:.2e}  time: {elapsed:.1f}s")

    if early_stop.early_stop:
        logger.save_weights(model.state_dict(), 'unet')
        print("Early stopping triggered.")
        break

logger.close()
print("End of training!")

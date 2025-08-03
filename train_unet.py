from torch.utils.data import DataLoader
from progress.bar import IncrementalBar
from network import *
from utils import *
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
import os, glob, json,time, torch




args = {
    "epochs": 1000,
    "batch_size": 128,
    "lr": 0.0005,
    "im_size": 256,
    "filts": 64,
    "num_images": 2400,
    "data_path": '/home/spm061102/Documents/TDG/Dataset/Emojis',
    "other": "40 MSE 60 SSIM, cells dataset, factor 0.5 and pat 20, tanh as output function and 0.5 Dropout"
}

logger = Logger(filename="Hybrid_loss1_cells")
logger.historical(args)


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
im_size = args["im_size"]
transforms = Compose([Resize((im_size,im_size)),
                      ToTensor()])

unet = Unet(filters = args["filts"]).to(device)
unet.apply(initialize_weights)

# Optimizers
optimizer = torch.optim.Adam(unet.parameters(), lr=args["lr"], betas=(0.5, 0.999))

# Loss
u_crit = HybLoss(lmb1 = 0.4, lmb2 = 0.6).to(device)

# Dataset
dataset = Data(root=args["data_path"], transform=transforms, num_images = args["num_images"], mode = 'train')
dataset_val = Data(root=args["data_path"], transform=transforms, num_images = args["num_images"], mode = 'val')

dataloader = DataLoader(dataset, batch_size=args["batch_size"], shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=args["batch_size"], shuffle=True)
print(len(dataloader))

scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)

print('Start of training process!')
for epoch in range(args["epochs"]):
    train_loss=0.
    val_loss=0.
    print(optimizer.param_groups[0]['lr'])

    start = time.time()
    bar = IncrementalBar(f'[Epoch {epoch+1}/{args["epochs"]}]', max=len(dataloader))
    unet.train()
    for src, tar in dataloader:
        src = src.to(device)
        tar = tar.to(device)

        # Unet training loss
        pred = unet(src)
        loss = u_crit(pred, tar)

        # Generator`s params update
        optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(unet.parameters())
        
        optimizer.step()

        # add batch losses
        train_loss += loss.item()
        bar.next()
    
    unet.eval()
    with torch.no_grad():
        for src, tar in dataloader_val:
            src = src.to(device)
            tar = tar.to(device)

            # Unet inference and loss for validation
            pred = unet(src)
            loss = u_crit(pred, tar)

        # add batch losses
            val_loss += loss.item()
    bar.finish()
    # obtain per epoch losses
    train_loss = train_loss/len(dataloader)
    val_loss = val_loss/len(dataloader_val)

    scheduler.step(val_loss)

    # count timeframe
    end = time.time()
    tm = (end - start)
    logger.add_scalar('train_loss', train_loss, epoch+1)
    logger.add_scalar('val_loss', val_loss, epoch+1)

    logger.save_weights(unet.state_dict(), 'unet')
    print("[Epoch %d/%d] [Train loss: %.3f][Val loss: %.3f]  ETA: %.3fs" % (epoch+1, args["epochs"], train_loss, val_loss, tm))

logger.historical(args)
logger.close()
print('End of training process!')
import numpy as numpy
import cv2 as cv
from network import *
from utils import *
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
import json



path_model = "/home/spm061102/Documents/TDG/models/July_22_2025_10_47PM_Hybrid_loss1_cells_unet.pt"
log_data = "/home/spm061102/Documents/TDG/models/July_22_2025_10_47PM_Hybrid_loss1_cells.json"

train_holo = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/src_ph/src0.png"
train_ph_gt = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/tar_ph/tar0.png"

val_holo = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/src_ph/src3400.png"
val_ph_gt = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/tar_ph/tar3400.png"

usaf_holo = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/src_ph/src3400.png"
usaf_ph_gt = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/tar_ph/tar3400.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Unet(filters=64)
model.load_state_dict(torch.load(path_model, map_location=torch.device(device), weights_only=True))


model.eval()


transforms = Compose([Resize((256,256)),
                      ToTensor()])


train_holo1 = Image.open(train_holo).convert('L')
train_ph_gt1 = Image.open(train_ph_gt).convert('L')

val_holo1 = Image.open(val_holo).convert('L')
val_ph_gt1 = Image.open(val_ph_gt).convert('L')

usaf_holo1 = Image.open(usaf_holo).convert('L')
usaf_ph_gt1 = Image.open(usaf_ph_gt).convert('L')


input_train = transforms(train_holo1).unsqueeze(0)
input_val = transforms(val_holo1).unsqueeze(0)
input_usaf = transforms(usaf_holo1).unsqueeze(0)


with torch.no_grad():
    output_train = model(input_train)

with torch.no_grad():
    output_val = model(input_val)

with torch.no_grad():
    output_usaf = model(input_usaf)

output_train = output_train[0,0,:,:]
output_val = output_val[0,0,:,:]
output_usaf = output_usaf[0,0,:,:]

fig, axes = plt.subplots(nrows=2, ncols=3)

axes[0, 0].imshow(train_ph_gt1, cmap='gray')
axes[0, 0].set_title('Train') # Optional: Add titles to subplots
axes[0, 0].axis('off') # Optional: Turn off axes ticks and labels

axes[1, 0].imshow(output_train, cmap='gray')
axes[1, 0].axis('off') # Optional: Turn off axes ticks and labels


axes[0, 1].imshow(val_ph_gt1, cmap='gray')
axes[0, 1].set_title('Validation') # Optional: Add titles to subplots
axes[0, 1].axis('off') # Optional: Turn off axes ticks and labels

axes[1, 1].imshow(output_val, cmap='gray')
axes[1, 1].axis('off') # Optional: Turn off axes ticks and labels


axes[0, 2].imshow(usaf_ph_gt1, cmap='gray')
axes[0, 2].set_title('Usaf') # Optional: Add titles to subplots
axes[0, 2].axis('off') # Optional: Turn off axes ticks and labels

axes[1, 2].imshow(output_usaf, cmap='gray')
axes[1, 2].axis('off') # Optional: Turn off axes ticks and labels

#plt.show()


with open(f'{log_data}', 'r') as file:
    data = json.load(file)


y_train = np.array(data["train_loss"])
x_train = np.shape(y_train)

y_val = data["val_loss"]
x_val = len(y_val)





print(y_train)

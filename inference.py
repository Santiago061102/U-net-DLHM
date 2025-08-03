from network import *
from utils import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch, json, cv2



'''
A code to make inference using the U-Net models trained.
Three images are load, one from train dataset, one from validation and 
a usaf that have not been seen by the network.
Also the loss function is visualized which must be saved as json dict, 
with train and  validation loss.
'''

# Model and loss data paths
path_model = "/home/spm061102/Documents/TDG/models/July_22_2025_10_47PM_Hybrid_loss1_cells_unet.pt"
log_data = "/home/spm061102/Documents/TDG/models/July_23_2025_03_19AM_Hybrid_loss2_cells.json"

# Images for inference
train_holo = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/src_ph/src0.png"
train_ph_gt = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/tar_ph/tar0.png"

val_holo = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/src_ph/src3400.png"
val_ph_gt = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/tar_ph/tar3400.png"

usaf_holo = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/src_ph/src3400.png"
usaf_ph_gt = "/home/spm061102/Documents/TDG/Dataset/Cancer blood cells/tar_ph/tar3400.png"


# GPU o CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# U-Net model load, with the same structure of the one trained
model = Unet(filters=64)
model.load_state_dict(torch.load(path_model, map_location=torch.device(device), weights_only=True))

# Evaluation mode
model.eval()

# Resize tranformation
transforms = Compose([Resize((256,256)),
                      ToTensor()])


train_holo1 = Image.open(train_holo).convert('L')
train_ph_gt1 = Image.open(train_ph_gt).convert('L')

val_holo1 = Image.open(val_holo).convert('L')
val_ph_gt1 = Image.open(val_ph_gt).convert('L')

usaf_holo1 = Image.open(usaf_holo).convert('L')
usaf_ph_gt1 = Image.open(usaf_ph_gt).convert('L')

# Transformation apply
input_train = transforms(train_holo1).unsqueeze(0)
input_val = transforms(val_holo1).unsqueeze(0)
input_usaf = transforms(usaf_holo1).unsqueeze(0)



# Inference 
with torch.no_grad():
    output_train = model(input_train)

with torch.no_grad():
    output_val = model(input_val)

with torch.no_grad():
    output_usaf = model(input_usaf)

# To CPU and numpy, must be rotate
output_train = output_train[0,0,:,:].cpu().numpy()
output_train = cv2.rotate(cv2.flip(output_train, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)


output_val = output_val[0,0,:,:].cpu().numpy()
output_val = cv2.rotate(cv2.flip(output_val, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)

output_usaf = output_usaf[0,0,:,:].cpu().numpy()
output_usaf = cv2.rotate(cv2.flip(output_usaf, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)


# Visualization of the inferences
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

plt.show()


# Data loss load
data = json.load(open(f'{log_data}', 'r'))
data_train = data["train_loss"]
data_val = data["val_loss"]


# Train loss
x_train = [key for key, _ in data_train.items()]
y_train = [value for _, value in data_train.items()]

# Validation loss
x_val = [key for key, _ in data_train.items()]
y_val = [value for _, value in data_val.items()]


plt.plot(x_train,y_train, color='maroon')
plt.plot(x_val,y_val, color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0, len(x_train)+1, step = (len(x_train))/10)-1)
plt.show()
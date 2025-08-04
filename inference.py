from network import *
from utils import *
from PIL import Image
from skimage.metrics import structural_similarity as ssim
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
path_model = r'/home/spm061102/Documents/TDG/others/August_03_2025_05_07PM_Hybrid_loss1_cells_unet.pt'
log_data = r'/home/spm061102/Documents/TDG/others/August_03_2025_05_07PM_Hybrid_loss1_cells.json'

# Images for inference
train_holo = r'/home/spm061102/Documents/TDG/Dataset/White blood cells/src/src0.png'
train_ph_gt = r'/home/spm061102/Documents/TDG/Dataset/White blood cells/tar_ph/tar0.png'

val_holo = r'/home/spm061102/Documents/TDG/Dataset/Wheat/src/src300.png'
val_ph_gt = r'/home/spm061102/Documents/TDG/Dataset/Wheat/tar_ph/tar300.png'

test_holo = r'/home/spm061102/Documents/TDG/Dataset/Rand figs/src/src400.png'
test_ph_gt = r'/home/spm061102/Documents/TDG/Dataset/Rand figs/tar_ph/tar_ph400.png'


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
train_ph_gt1 = np.array(Image.open(train_ph_gt).convert('L'))
train_ph_gt1 = (train_ph_gt1 - np.min(train_ph_gt1))/(np.max(train_ph_gt1) - np.min(train_ph_gt1))

val_holo1 = Image.open(val_holo).convert('L')
val_ph_gt1 = np.array(Image.open(val_ph_gt).convert('L'))
val_ph_gt1 = (val_ph_gt1 - np.min(val_ph_gt1))/(np.max(val_ph_gt1) - np.min(val_ph_gt1))

test_holo1 = Image.open(test_holo).convert('L')
test_ph_gt1 = np.array(Image.open(test_ph_gt).convert('L'))
test_ph_gt1 = (test_ph_gt1 - np.min(test_ph_gt1))/(np.max(test_ph_gt1) - np.min(test_ph_gt1))

# Transformation apply
input_train = transforms(train_holo1).unsqueeze(0)
input_val = transforms(val_holo1).unsqueeze(0)
input_test = transforms(test_holo1).unsqueeze(0)



# Inference 
with torch.no_grad():
    output_train = model(input_train)

with torch.no_grad():
    output_val = model(input_val)

with torch.no_grad():
    output_test = model(input_test)

# To CPU and numpy, must be rotate and normalized
output_train = output_train[0,0,:,:].cpu().numpy()
output_train = cv2.rotate(cv2.flip(output_train, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
output_train = (output_train - np.min(output_train))/(np.max(output_train) - np.min(output_train))

output_val = output_val[0,0,:,:].cpu().numpy()
output_val = cv2.rotate(cv2.flip(output_val, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
output_val = (output_val - np.min(output_val))/(np.max(output_val) - np.min(output_val))

output_test = output_test[0,0,:,:].cpu().numpy()
output_test = cv2.rotate(cv2.flip(output_test, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
output_test = (output_test - np.min(output_test))/(np.max(output_test) - np.min(output_test))

# The MSE of the train, validation and test images are calculated and displayed
mse_train = np.mean((train_ph_gt1 - output_train)**2)
mse_val = np.mean((val_ph_gt1 - output_val)**2)
mse_test = np.mean((test_ph_gt1 - output_test)**2)


# The SSIM of the train, validation and test images are calculated and displayed
(ssim_train, diff) = ssim(train_ph_gt1, output_train, full=True, data_range=train_ph_gt1.max() - train_ph_gt1.min())
(ssim_val, diff) = ssim(val_ph_gt1, output_val, full=True, data_range=val_ph_gt1.max() - val_ph_gt1.min())
(ssim_test, diff) = ssim(test_ph_gt1, output_test, full=True, data_range=test_ph_gt1.max() - test_ph_gt1.min())

print(f'MSE: Train {mse_train}, Validation {mse_val}, Test {mse_test}')
print(f'SSIM: Train {ssim_train}, Validation {ssim_val}, Test {ssim_test}')


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


axes[0, 2].imshow(test_ph_gt1, cmap='gray')
axes[0, 2].set_title('Usaf') # Optional: Add titles to subplots
axes[0, 2].axis('off') # Optional: Turn off axes ticks and labels

axes[1, 2].imshow(output_test, cmap='gray')
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
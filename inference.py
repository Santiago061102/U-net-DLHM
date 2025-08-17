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
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
# Model and loss data paths
pathModel = r'/home/spm061102/Documents/TDG/models/models/August_15_2025_12_07AM_Rand_figs1_beta1_unet.pt'
logData = r'/home/spm061102/Documents/TDG/models/models/August_15_2025_12_07AM_Rand_figs1_beta1.json'

# Images for inference
pathTrainHolo = r'/home/spm061102/Documents/TDG/Dataset/random shapes/src/src0.png'
pathTrainPhGT = r'/home/spm061102/Documents/TDG/Dataset/random shapes/tar_ph/tar_ph0.png'

pathValHolo = r'/home/spm061102/Documents/TDG/Dataset/white blood cells/src/src0.png'
pathValPhGT = r'/home/spm061102/Documents/TDG/Dataset/white blood cells/tar_ph/tar0.png'

pathTestHolo = r'/home/spm061102/Documents/TDG/Dataset/test/src/src0.png'
pathTestPhGT = r'/home/spm061102/Documents/TDG/Dataset/test/tar_ph/tar_ph0.png'


# GPU o CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device) 


# U-Net model load, with the same structure of the one trained
model = UNet(filters=64)
model.load_state_dict(torch.load(pathModel, map_location=torch.device(device), weights_only=True))

# Evaluation mode
model.eval()

imgSize = 256
# Resize tranformation
transforms = Compose([Resize((imgSize, imgSize)),
                      ToTensor()])


trainHolo = Image.open(pathTrainHolo).convert('L')
pathTrainPhGT = Image.open(pathTrainPhGT).convert('L')

valHolo = Image.open(pathValHolo).convert('L')
pathValPhGT = Image.open(pathValPhGT).convert('L')

testHolo = Image.open(pathTestHolo).convert('L')
pathTestPhGT = Image.open(pathTestPhGT).convert('L')

# Transformation apply
inputTrain = transforms(trainHolo).unsqueeze(0)
inputVal = transforms(valHolo).unsqueeze(0)
inputTest = transforms(testHolo).unsqueeze(0)

trainPhGT = transforms(pathTrainPhGT).unsqueeze(0)
valPhGT = transforms(pathValPhGT).unsqueeze(0)
testPhGT = transforms(pathTestPhGT).unsqueeze(0)



# Inference 
with torch.no_grad():
    outputTrain = model(inputTrain)

with torch.no_grad():
    outputVal = model(inputVal)

with torch.no_grad():
    outputTest = model(inputTest)

crit = SSIMLoss()
ssimTrain = crit(trainPhGT, outputTrain)
ssimVal = crit(valPhGT, outputVal)
ssimTest = crit(testPhGT, outputTest)

# To CPU and numpy, must be rotate and normalized
outputTrain = outputTrain[0,0,:,:].cpu().numpy()
outputTrain = cv2.rotate(cv2.flip(outputTrain, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)

outputVal = outputVal[0,0,:,:].cpu().numpy()
outputVal = cv2.rotate(cv2.flip(outputVal, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)

outputTest = outputTest[0,0,:,:].cpu().numpy()
outputTest = cv2.rotate(cv2.flip(outputTest, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)

# The MSE of the train, validation and test images are calculated and displayed
trainPhGT = trainPhGT[0,0,:,:].numpy()
valPhGT = valPhGT[0,0,:,:].numpy()
testPhGT = testPhGT[0,0,:,:].numpy()

mseTrain = np.mean((trainPhGT- outputTrain)**2)
mseVal = np.mean((valPhGT - outputVal)**2)
mseTest = np.mean((testPhGT - outputTest)**2)


# The SSIM of the train, validation and test images are calculated and displayed

print(f'MSE: Train {mseTrain:.5f}, Validation {mseVal:5f}, Test {mseTest:5f}')
print(f'SSIM: Train {ssimTrain:.5f}, Validation {ssimVal:.5f}, Test {ssimTest:.5f}')



# Visualization of the inferences
fig, axes = plt.subplots(nrows=2, ncols=3)

axes[0, 0].imshow(trainPhGT, cmap='gray')
axes[0, 0].set_title('Train') # Optional: Add titles to subplots
axes[0, 0].axis('off') # Optional: Turn off axes ticks and labels

axes[1, 0].imshow(outputTrain, cmap='gray')
axes[1, 0].axis('off') # Optional: Turn off axes ticks and labels


axes[0, 1].imshow(valPhGT, cmap='gray')
axes[0, 1].set_title('Validation') # Optional: Add titles to subplots
axes[0, 1].axis('off') # Optional: Turn off axes ticks and labels

axes[1, 1].imshow(outputVal, cmap='gray')
axes[1, 1].axis('off') # Optional: Turn off axes ticks and labels


axes[0, 2].imshow(testPhGT, cmap='gray')
axes[0, 2].set_title('Test') # Optional: Add titles to subplots
axes[0, 2].axis('off') # Optional: Turn off axes ticks and labels

axes[1, 2].imshow(outputTest, cmap='gray')
axes[1, 2].axis('off') # Optional: Turn off axes ticks and labels

plt.show()


# Data loss load
data = json.load(open(f'{logData}', 'r'))
dataTrain = data["train_loss"]
dataVal = data["val_loss"]


# Train loss
xTrain = [key for key, _ in dataTrain.items()]
yTrain = [value for _, value in dataTrain.items()]

# Validation loss
xVal = [key for key, _ in dataVal.items()]
yVal = [value for _, value in dataVal.items()]


plt.plot(xTrain,yTrain, color='maroon')
plt.plot(xVal,yVal, color='green')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.xticks(np.arange(0, len(xTrain)+1, step = (len(xTrain))/10)-1)
plt.show()
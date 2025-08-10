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
pathModel = r'/home/spm061102/Documents/TDG/models/models/August_07_2025_03_32PM_Comb512_unet.pt'
logData = r'/home/spm061102/Documents/TDG/models/models/August_07_2025_03_32PM_Comb512.json'

# Images for inference
pathTrainHolo = r'/home/spm061102/Documents/TDG/Dataset/white blood cells/src/src0.png'
pathTrainPhGT = r'/home/spm061102/Documents/TDG/Dataset/white blood cells/tar_ph/tar0.png'

pathValHolo = r'/home/spm061102/Documents/TDG/Dataset/emojis/src/src300.png'
pathValPhGT = r'/home/spm061102/Documents/TDG/Dataset/emojis/tar_ph/tar_ph300.png'

pathTestHolo = r'/home/spm061102/Documents/TDG/Dataset/test/src/src0.png'
pathTestPhGT = r'/home/spm061102/Documents/TDG/Dataset/test/tar_ph/tar_ph0.png'


# GPU o CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# U-Net model load, with the same structure of the one trained
model = Unet(filters=64)
model.load_state_dict(torch.load(pathModel, map_location=torch.device(device), weights_only=True))

# Evaluation mode
model.eval()

imgSize = 512
# Resize tranformation
transforms = Compose([Resize((imgSize, imgSize)),
                      ToTensor()])


trainHolo = Image.open(pathTrainHolo).convert('L')
trainPhGT = cv2.resize(np.array(Image.open(pathTrainPhGT).convert('L')), (imgSize, imgSize))
trainPhGT = (trainPhGT - np.min(trainPhGT))/(np.max(trainPhGT) - np.min(trainPhGT))

valHolo = Image.open(pathValHolo).convert('L')
valPhGT = cv2.resize(np.array(Image.open(pathValPhGT).convert('L')), (imgSize, imgSize))
valPhGT = (valPhGT - np.min(valPhGT))/(np.max(valPhGT) - np.min(valPhGT))

testHolo = Image.open(pathTestHolo).convert('L')
testPhGT = cv2.resize(np.array(Image.open(pathTestPhGT).convert('L')), (imgSize, imgSize))
testPhGT = (testPhGT - np.min(testPhGT))/(np.max(testPhGT) - np.min(testPhGT))

# Transformation apply
input_train = transforms(trainHolo).unsqueeze(0)
input_val = transforms(valHolo).unsqueeze(0)
input_test = transforms(testHolo).unsqueeze(0)



# Inference 
with torch.no_grad():
    outputTrain = model(input_train)

with torch.no_grad():
    outputVal = model(input_val)

with torch.no_grad():
    outputTest = model(input_test)

# To CPU and numpy, must be rotate and normalized
outputTrain = outputTrain[0,0,:,:].cpu().numpy()
outputTrain = cv2.rotate(cv2.flip(outputTrain, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
outputTrain = (outputTrain - np.min(outputTrain))/(np.max(outputTrain) - np.min(outputTrain))

outputVal = outputVal[0,0,:,:].cpu().numpy()
outputVal = cv2.rotate(cv2.flip(outputVal, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
outputVal = (outputVal - np.min(outputVal))/(np.max(outputVal) - np.min(outputVal))

outputTest = outputTest[0,0,:,:].cpu().numpy()
outputTest = cv2.rotate(cv2.flip(outputTest, 1), cv2.ROTATE_90_COUNTERCLOCKWISE)
outputTest = (outputTest - np.min(outputTest))/(np.max(outputTest) - np.min(outputTest))

# The MSE of the train, validation and test images are calculated and displayed
mseTrain = np.mean((trainPhGT - outputTrain)**2)
mseVal = np.mean((valPhGT - outputVal)**2)
mseTest = np.mean((testPhGT - outputTest)**2)


# The SSIM of the train, validation and test images are calculated and displayed
(ssimTrain, diff) = ssim(trainPhGT, outputTrain, full=True, data_range=trainPhGT.max() - trainPhGT.min())
(ssimVal, diff) = ssim(valPhGT, outputVal, full=True, data_range=valPhGT.max() - valPhGT.min())
(ssimTest, diff) = ssim(testPhGT, outputTest, full=True, data_range=testPhGT.max() - testPhGT.min())

print(f'MSE: Train {mseTrain}, Validation {mseVal}, Test {mseTest}')
print(f'SSIM: Train {ssimTrain}, Validation {ssimVal}, Test {ssimTest}')


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
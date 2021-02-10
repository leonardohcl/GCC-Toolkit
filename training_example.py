import torch.nn as nn
import torch.optim as optim
from torchvision import models,transforms
import Dataset as dataset

# 1. Load model
net = models.resnet50(pretrained=True)

# 2. Freeze the training for all layers
# Obs. This step only applies for transfer learning, if it's not your case just ignore it
for param in net.parameters():
    param.requires_grad = False

# 3. Update output to match number of classes
num_feats = net.fc.in_features
net.fc = nn.Linear(num_feats, 2)

# 4. Create transforms for the data
# Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
# ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
# using this values, but hey can be changed to values that best suits your case. 
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 5. Create dataset. This type can be found in the file Dataset.py of this package 
# and gets the path to a csv with the list of the images file names and the base path to the folder of the
# images. If you don't have the csv already, you can use the 'createFolderContentCsv' function
# from the file FileHandling.py. 
data = dataset.ImageDataset("csv_path", "images_folder_path", 2, transform=trans)

# 6. Specify error and optimization functions 
crit = nn.MSELoss()
opt = optim.SGD(net.parameters(), lr=0.001)

# 7. Call the training function
trainedModel = cnn.trainCrossValidation(net,data,5,crit,opt,50, plotAcc = True)
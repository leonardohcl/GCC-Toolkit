import torch.nn as nn
import torch.optim as optim
from torchvision import models,transforms
import Dataset as dataset
import CNN

# 0. Define important variables
# Input detais
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample"
NUMBER_OF_CLASSES = 2
LIST_OF_CLASSES = [0, 1]
# Training details
LEARNING_RATE = 0.001
NUMBER_OF_FOLDS = 2
EPOCHS = 10
# Training Visualization
PLOT_ACCURACY = True
PLOT_LOSS = False


# 1. Load model
net = models.resnet50(pretrained=True)

# 2. Freeze the training for all layers
# Obs. This step only applies for transfer learning, if it's not your case just ignore it
for param in net.parameters():
    param.requires_grad = False

# 3. Update output to match number of classes
num_feats = net.fc.in_features
net.fc = nn.Linear(num_feats, NUMBER_OF_CLASSES)

# 4. Create transforms for the data
# Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
# ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
# using this values, but hey can be changed to values that best suits your case. 
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 5. Create dataset. This type can be found in the file Dataset.py of this package 
# and gets the path to a csv with the list of the images file names and the base path to the folder of the
# images. If you don't have the csv already, you can use the 'createFolderContentCsv' function
# from the file FileHandling.py. 
data = dataset.ImageDataset(IMAGE_CSV_PATH, IMAGE_FOLDER_PATH, NUMBER_OF_CLASSES, transform=trans)

# 6. Specify error and optimization functions 
ERROR_FUNCTION = nn.MSELoss()
OPTIMIZATION_FUNCTION = optim.SGD(net.parameters(), lr=LEARNING_RATE)

# 7. Call the training function
trainedModel = CNN.trainCrossValidation(net, data, NUMBER_OF_FOLDS, ERROR_FUNCTION, OPTIMIZATION_FUNCTION, EPOCHS, PLOT_ACCURACY, PLOT_LOSS )
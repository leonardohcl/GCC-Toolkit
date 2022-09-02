import torch
import torch.nn as nn
from torchvision import models, transforms
from Dataset import ImageDataset
from MachineLearning import Trainer

# 0. Define important variables
# Input detais
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample"
CLASSES = [0, 1]
# Training details
LEARNING_RATE = 0.001
LEARNING_RATE_DROP = 0.75
LEARNING_RATE_DROP_EVERY_N_EPOCHS = 2
NUMBER_OF_FOLDS = 2
EPOCHS = 10
# Training Visualization
PLOT_ACCURACY = True
PLOT_LOSS = False


# 1. Load model
# net = models.resnet50(pretrained=True)
# net = models.densenet121(pretrained=True)
model = models.efficientnet_b2(pretrained=True)

# 2. Freeze the training for all layers
# Obs. This step only applies for transfer learning, if it's not your case just ignore it
for param in model.parameters():
    param.requires_grad = False

# 3. Update output to match number of classes
# [resnet uses this values]
# num_feats = net.fc.in_features
# net.fc = nn.Linear(num_feats, len(LIST_OF_CLASSES))

# [densenet uses this values]
# num_feats = net.classifier.in_features
# net.classifier = nn.Linear(num_feats, len(LIST_OF_CLASSES))

# [efficientnet_b2 uses this values]
num_feats = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_feats, len(CLASSES))


# 4. Create transforms for the data
# Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
# ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
# using this values, but hey can be changed to values that best suits your case.
transform_functions = transforms.Compose([transforms.ToTensor(
), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 5. Create dataset. This type can be found in the file Dataset.py of this package
# and gets the path to a csv with the list of the images file names and the base path to the folder of the
# images. If you don't have the csv already, you can use the 'createFolderContentCsv' function
# from the file FileHandling.py.
dataset = ImageDataset(
    IMAGE_CSV_PATH,
    IMAGE_FOLDER_PATH,
    CLASSES,
    transform=transform_functions,
)

# 6. Call the training function
trained_model, learning_history = Trainer.k_fold_training(
    model,
    dataset,
    k=NUMBER_OF_FOLDS,
    epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    learning_rate_drop=LEARNING_RATE_DROP,  # optional
    learning_rate_drop_step_size=LEARNING_RATE_DROP_EVERY_N_EPOCHS,  # optional
    plot_acc=PLOT_ACCURACY,  # optional
    plot_loss=PLOT_LOSS,  # optional
    error_fn=nn.CrossEntropyLoss(),  # optional
    # max_batch_size=25 # optional, if you want to limit how many images are loaded to the memory at once. By default there's no limit
    # log_filename="training_example", # optional, if you want to save the log of the training
    # is_notebook_env=True, #optional, set to True if using a python notebook (handles output messages better in this cases) 
)

# 7. Save trained model (Optional)
torch.save(trained_model.state_dict(), "trained_model")

import os
import torch
import Dataset as ds
import CAMHelpers as cam
import FileHandling as fh
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM

# 0. Define important variables
# Input details
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample"
NUMBER_OF_CLASSES = 2

# Output details
OUTPUT_FOLDER = "sample-CAM"
SAVE_BW_CAM = True
SAVE_BW_OVERLAY_CAM = True
SAVE_JET_COLORMAP_CAM = True
SAVE_JET_COLORMAP_OVERLAY_CAM = True

# Output details for more than one type of output
BW_FOLDER_NAME = os.path.join(OUTPUT_FOLDER, "Default")
BW_OVERLAY_FOLDER_NAME = os.path.join(OUTPUT_FOLDER, "Overlay")
JET_FOLDER_NAME = os.path.join(OUTPUT_FOLDER, "Colormap")
JET_OVERLAY_FOLDER_NAME = os.path.join(OUTPUT_FOLDER, "Colormap Overlay")

# 0.1. Just to be the shure the output folder(s) exists
print("Setting up extractors and structures... please hold on :)")
if os.path.isdir(OUTPUT_FOLDER) == False:
    os.mkdir(OUTPUT_FOLDER)

# 0.2 Arrangements to save multiple CAMs in different foldes
output_types = SAVE_BW_CAM + SAVE_BW_OVERLAY_CAM + SAVE_JET_COLORMAP_CAM + SAVE_JET_COLORMAP_OVERLAY_CAM
if output_types > 1:
    if SAVE_BW_CAM:
        if os.path.isdir(BW_FOLDER_NAME) == False:
            os.mkdir(BW_FOLDER_NAME)

    if SAVE_BW_OVERLAY_CAM:
        if os.path.isdir(BW_OVERLAY_FOLDER_NAME) == False:
            os.mkdir(BW_OVERLAY_FOLDER_NAME)

    if SAVE_JET_COLORMAP_CAM:
        if os.path.isdir(JET_FOLDER_NAME) == False:
            os.mkdir(JET_FOLDER_NAME)

    if SAVE_JET_COLORMAP_OVERLAY_CAM:
        if os.path.isdir(JET_OVERLAY_FOLDER_NAME) == False:
            os.mkdir(JET_OVERLAY_FOLDER_NAME)


# 1. Get the CNN model
model = models.vgg16(pretrained=True)
model.eval()  # set it to evaluation mode

# 1.1. Define the target layer, usually it's last convolutional on the model
# For VGG19 (and others VGG architectures) it's the 'features[-1]'
# For ResNet50 it's the 'layer4[-1]'
# For InceptionV3 it's the 'Mixed_7c'
target_layer = model.features[-1]

# 1.2. Create CAM generator
cam_generator = GradCAM(model=model, target_layer=target_layer, use_cuda = torch.cuda.is_available())

# 2. Define the dataset
# Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
# ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
# using this values, but hey can be changed to values that best suits your case.
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize([
                           0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = ds.ImageDataset(
    IMAGE_CSV_PATH, IMAGE_FOLDER_PATH, NUMBER_OF_CLASSES, trans)
img_count = len(dataset)

# 3. Process the dataset
for idx in range(img_count):
    filename = dataset.getFilename(idx)
    print("processing ({}/{})... {}".format(idx + 1, img_count ,filename))

    # Get image input
    img = dataset[idx].unsqueeze(0)

    # Get the CAM
    output = cam_generator(input_tensor=img)

    # Create the image
    output_img = cam.createImage(output) # this creates a black and white map 
    output_colormap_img = cam.createImage(output, "jet") # this create a map with a colormap

    original_img = fh.readImage(dataset.getFilepath(idx)) # this gets the original image, used when overlaying
    filename = filename.split(".")[0]
    # Save it
    if SAVE_BW_CAM:
        target_folder = OUTPUT_FOLDER if output_types == 1 else BW_FOLDER_NAME
        fh.saveImage(output_img, target_folder, filename)

    if SAVE_BW_OVERLAY_CAM:
        target_folder = OUTPUT_FOLDER if output_types == 1 else BW_OVERLAY_FOLDER_NAME
        fh.saveImage(cam.multiplyCAM(original_img, output_img), target_folder, filename)

    if SAVE_JET_COLORMAP_CAM:
        target_folder = OUTPUT_FOLDER if output_types == 1 else JET_FOLDER_NAME
        fh.saveImage(output_colormap_img, target_folder, filename)

    if SAVE_JET_COLORMAP_OVERLAY_CAM:
        target_folder = OUTPUT_FOLDER if output_types == 1 else JET_OVERLAY_FOLDER_NAME
        fh.saveImage(cam.overlayCAM(original_img, output_colormap_img), target_folder, filename)

import os
import torch
import numpy as np
from Dataset import ImageDataset
from File import ImageFile
from lime import lime_image
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# 0. Define important variables
# Input info
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample/images"
CLASSES = [0, 1]

# Output info
OUTPUT_FOLDER = "sample-LIME"

# LIME info
N_PERTURBATIONS = 1000  # number of perturbations used to get the LIME explanation

# 0.1. Make sure the output folder exists
if os.path.isdir(OUTPUT_FOLDER) == False:
    os.mkdir(OUTPUT_FOLDER)

# 1. Define the dataset
# Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
# ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
# using this values, but hey can be changed to values that best suits your case.
transform_functions = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
dataset = ImageDataset(IMAGE_CSV_PATH,
                       IMAGE_FOLDER_PATH,
                       CLASSES,
                       transform_functions)
img_count = len(dataset)

# 2. Get the CNN model
model = models.resnet50(pretrained=True)
model.eval()  # set it to evaluation mode

# 3. Define a function for the CNN to classify inputs (required later for the LIME explainer)


def batch_predict(images):
    # transform input images to tensors
    batch = torch.stack(tuple(transform_functions(i) for i in images), dim=0)

    # use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    # get ouput
    logits = model(batch)

    # now place the outputs as values between 1 and 0, so later
    # the explainer can deal with it as probabilities instead of
    # a plain number output
    probs = F.softmax(logits, dim=1)

    # return the probabilitieas as a numpy array
    return probs.detach().cpu().numpy()


# 4. create the LIME explainer
explainer = lime_image.LimeImageExplainer()

# 3. Process the dataset
# obs. tqdm here is just so there's a progressbar showing the loop progress
for image_index in range(img_count):
    filename = dataset[image_index].filename
    print(f"processing {filename} ({image_index + 1}/{img_count})")
    img = dataset[image_index].get_image()  # load image as is
    img = img.convert("RGB")

    # get the LIME explanation for the image
    explanation = explainer.explain_instance(np.array(img),
                                             batch_predict,  # classification function created before
                                             top_labels=1,  # the amount of labes considered on explaination, for now we only want the classification so the highest probability output
                                             hide_color=0,  # color to use when hiding superpixels, if none set it would use the average for the channel on the image so we better put 0 to use black
                                             num_samples=N_PERTURBATIONS
                                             )

    # get the binary mask that defines where are the superpixels that explain the classification
    temp_img, mask = explanation.get_image_and_mask(explanation.top_labels[0],
                                                    positive_only=True,  # this is a flag to indicate that we only wnat the superpixels that have contributed to the classification, there's also a negative_only parameter that shows the pixels that have had a negative contribution to the classification
                                                    num_features=5,  # number of superpixels to be used to explain the classfication, 5 is the default number but feel free to move it around
                                                    )

    # apply mask to the image
    width, height = np.shape(mask)
    for i in range(height):
        for j in range(width):
            if (mask[j][i] == 0):
                temp_img[:][j][i] = 0

    lime_img = Image.fromarray(temp_img)

    # save the LIME image
    filename = filename.split('.')[0]
    ImageFile.save(lime_img, OUTPUT_FOLDER, filename)

    if (torch.cuda.is_available()):
        torch.cuda.empty_cache()  # It's important to clear the cuda memory of unused data after getting a LIME from an image. This algorithms takes a lot of the memory available and if processing several images there probably would be an out of memory error if this was not done

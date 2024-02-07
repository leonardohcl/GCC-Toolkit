import torch
import numpy as np
from File import Arff
from Dataset import ImageDataset
from torchvision import models, transforms
from tqdm import tqdm

# 0. Define important variables
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample/images"
CLASSES = [0, 1]
OUTPUT_NAME = "sample-extraction-data"
HOLDER_NAME = 'deep_feats'
USE_GPU = True

# 1. Create dictionary to hold outputs
data_holder = {}

# 2. Define function to send input/output to holder with a specific name
def get_input(name):
    def hook(model, input, output):
        aux_array = input[0].cpu().detach().numpy()
        aux_array = aux_array.flatten()
        data_holder[name] = aux_array.tolist()

    return hook

def get_output(name):
    def hook(model, input, output):
        aux_array = output.cpu().detach().numpy()
        aux_array = aux_array.flatten()
        data_holder[name] = aux_array.tolist()

    return hook


# 3. Load database
# Obs. Normalization is encouraged if using a pretrained model, the values correspond to the
# ImageNet dataset mean and standard deviations of each color channel. The pretraining was applied
# using this values, but hey can be changed to values that best suits your case.
transform_functions = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
dataset = ImageDataset(IMAGE_CSV_PATH,
                       IMAGE_FOLDER_PATH,
                       CLASSES,
                       transform=transform_functions)

img_count = len(dataset)  # get image count

# 4. Load CNN
net = models.resnet50(pretrained=True)
net.eval()  # set it to evaluation mode

# 4.1. If GPU is available, send cnn to it
if USE_GPU and torch.cuda.is_available():
    net.cuda()

# For densenet121 use
# output_size = net.classifier.in_features # or 1024
# For efficientnet_b2 use # or 1408
# output_size = net.classifier[1].in_features 
# For inception_v3 and resnet50 use 
output_size = net.fc.in_features  # or 2048
# For vgg19 use
# output_size = net.classifier[6].in_features # or 4096

# 5. Register hook to desired layer to hold it's output
# For densenet121 use
# net.classifier.register_forward_hook(get_input(HOLDER_NAME))
# For efficientnet_b2, inception_v3 and resnet50 use
net.avgpool.register_forward_hook(get_output(HOLDER_NAME))
# For vgg19 use
# net.classifier[5].register_forward_hook(get_output(HOLDER_NAME))


# 6. Process each image and get the desired layers output
# create placeholder output list
output = np.zeros((img_count, output_size + 1)).tolist()
progress = tqdm(range(img_count))
for idx in progress:
    progress.set_description(dataset[idx].filename)

    # create fake batch with single input
    img = dataset[idx].get_tensor().unsqueeze(0)
    # if cuda is available send input to it
    if USE_GPU and torch.cuda.is_available():
        img = img.cuda()

    # give input to cnn
    out = net(img)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # get all the data collected in the holder
    img_output = data_holder[HOLDER_NAME]
    # append image class to the end of the output
    img_output.append(dataset[idx].class_id)

    # store output
    output[idx] = img_output

# 7. Create the names for the extracted variables
names = ["avgpool{}".format(i) for i in range(1, output_size+1)]

# 8. Write arff file
data = Arff(relation=OUTPUT_NAME,
            entries=output,
            classes=CLASSES,
            attrs=names, attr_types=['numeric' for _ in names])
data.save(OUTPUT_NAME)

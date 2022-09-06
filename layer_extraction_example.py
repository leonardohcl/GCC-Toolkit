import torch
import time
import numpy as np
import File as fh
from Dataset import ImageDataset
from torchvision import models, transforms

# 0. Define important variables
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample"
CLASSES = [0, 1]
OUTPUT_NAME = "sample-extraction-data"

# 1. Create dictionary to hold outputs
output_holder = {}

# 2. Define function to send output to holder with a specific name


def get_output(layer_name):
    def hook(model, input, output):
        aux_array = output.cpu().detach().numpy()
        aux_array = aux_array.flatten()
        output_holder[layer_name] = aux_array.tolist()

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
if torch.cuda.is_available():
    net.cuda()

output_size = net.fc.in_features  # get output size
# For vgg19 use
# output_size = net.classifier[6].in_features

# 5. Register hook to desired layer to hold it's output
net.avgpool.register_forward_hook(get_output("avg_pool"))
# For vgg19 use
# net.classifier[5].register_forward_hook(getOutput("fc"))


# 6. Process each image and get the desired layers output
# create placeholder output list
output = np.zeros((img_count, output_size + 1)).tolist()
start = time.time()
for idx in range(img_count):
    print("processing ({}/{}): {}".format(idx +
          1, img_count, dataset[idx].filename))

    # create fake batch with single input
    img = dataset[idx].get_tensor().unsqueeze(0)
    # if cuda is available send input to it
    if torch.cuda.is_available():
        img = img.cuda()

    # give input to cnn
    out = net(img)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # get all the data collected in the holder
    img_output = output_holder["avg_pool"]
    # append image class to the end of the output
    img_output.append(dataset[idx].class_id)

    # store output
    output[idx] = img_output

# 7. Create the names for the extracted variables
names = ["avgpool{}".format(i) for i in range(1, output_size+1)]

# 8. Write arff file
fh.save(OUTPUT_NAME, output, names, CLASSES)

time_elapsed = time.time() - start
print('Extraction complete in {:.0f}m {:.0f}s'.format(
    time_elapsed // 60, time_elapsed % 60))

import os
import Fractal 
import pandas as pd

# 0. Define important variables
# Input details
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample"
PROB_MATRIX_JSON_PATH = "sample/go-probability-matrix.json"
MIN_R = 3
MAX_R = 5

# 1. Read image csv
IMAGE_LIST = pd.read_csv(IMAGE_CSV_PATH, header=None)

# 2. Process all images on the list to obtain its fractal properties
for idx in range(len(IMAGE_LIST)):
    # 2.1. Get image file name and path
    image_filename = IMAGE_LIST.iloc[idx, 0]
    image_path = os.path.join(IMAGE_FOLDER_PATH, image_filename)

    # 2.2. Processos image to obtain its probability matrix with the gliding box technique
    probMatrix = Fractal.GlidingBoxProbabilityMatrix(image_path, min_r=MIN_R ,max_r=MAX_R)

    # 2.2. Extract lacunarity from the probability matrix
    lac = Fractal.LacunarityFromProabilityMatrix(probMatrix,min_r=MIN_R ,max_r=MAX_R)

    # 2.3. Extract fractal dimension from the probability matrix
    fd = Fractal.FractalDimensionFromProabilityMatrix(probMatrix,min_r=MIN_R ,max_r=MAX_R)

    print(lac)
    print(fd)


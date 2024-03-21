import os
from Fractal import GlidingBox
import pandas as pd

# 0. Define important variables
# Input details
IMAGE_CSV_PATH = "sample/sample-images.csv"
IMAGE_FOLDER_PATH = "sample/images"
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
    prob_matrix = GlidingBox.probability_matrix(
        image_path, min_r=MIN_R, max_r=MAX_R)

    # 2.2. Extract lacunarity from the probability matrix
    lac = GlidingBox.lacunarity(prob_matrix, min_r=MIN_R, max_r=MAX_R)

    # 2.3. Extract fractal dimension from the probability matrix
    fd = GlidingBox.fractal_dimension(prob_matrix, min_r=MIN_R, max_r=MAX_R)

    perc = GlidingBox.percolation(image_path, min_r=MIN_R, max_r=MAX_R)
    print(f"{lac} <- Lacunarity")
    print(f"{fd} <- Fractal Dimension")
    print(f"{perc.avg_cluster_count} <- [p(r)] average cluster/box")
    print(f"{perc.avg_biggest_cluster_area} <- [h(r)] average biggest cluster area")
    print(f"{perc.percolation} <- [g(r)] percolation")

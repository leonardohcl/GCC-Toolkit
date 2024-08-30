import pandas as pd
from Fractal import PROBABILITY_MATRIX_ARFF_ATTRIBUTES
from File import Arff
import GoLangJson
from Curve import Curve
from tqdm import tqdm

# 0. Define important variables
# Input details
IMAGE_CSV_PATH = "sample/sample-images.csv"
PROB_MATRIX_JSON_PATH = "sample/go-probability-matrix.json"
MIN_R = 3
MAX_R = 41
R_RANGE = range(MIN_R, MAX_R + 1, 2)

# Arff definitions
ARFF_NAME = "test"
ARFF_CLASSES = [0, 1]
ARFF_CONTENT = []

# 1. Read image csv and probability matrix
IMAGE_LIST = pd.read_csv(IMAGE_CSV_PATH, header=None)
PROB_MATRIX = GoLangJson.ProbabilityMatrixJson(PROB_MATRIX_JSON_PATH)

# 2. Iterate through images to calculate the fractal properties
progress = tqdm(range(len(IMAGE_LIST)))
for idx in progress:
    # Get image name and class
    image_name = IMAGE_LIST.iloc[idx, 0]
    image_class = IMAGE_LIST.iloc[idx, 1]

    progress.set_description(image_name)    

    # get lacunarities and fractal dimensions
    lacunarity_curve = PROB_MATRIX.get_lacunarity_curve(image_name, MIN_R, MAX_R)
    fractal_dimensions = PROB_MATRIX.get_fractal_dimension_curve(image_name, MIN_R, MAX_R)
        
    
    features = lacunarity_curve + fractal_dimensions + Curve.get_descriptors(R_RANGE, lacunarity_curve) + [image_class]
    ARFF_CONTENT.append(features)

# 3. Create .arff file
arff = Arff(relation=ARFF_NAME,
            entries=ARFF_CONTENT,
            attrs=PROBABILITY_MATRIX_ARFF_ATTRIBUTES, 
            classes=ARFF_CLASSES)
arff.save(ARFF_NAME)

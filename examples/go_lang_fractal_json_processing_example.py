import pandas as pd
from File import Arff
import GoLangJson
from Curve import Curve
from tqdm import tqdm
from Fractal import PERCOLATION_ARFF_ATTRIBUTES, PROBABILITY_MATRIX_ARFF_ATTRIBUTES

# 0. Define important variables
# Input details
IMAGE_CSV_PATH = "sample/sample-images.csv"
PERCOLATION_JSON_PATH = "sample/go-percolation-data.json"
PROB_MATRIX_JSON_PATH = "sample/go-probability-matrix.json"
MIN_R = 3
MAX_R = 41
R_RANGE = range(MIN_R, MAX_R + 1, 2)

# Arff definitions
ARFF_NAME = "fractal-example"
ARFF_CLASSES = [0, 1]
ARFF_CONTENT = []


# 1. Read image csv and json files
IMAGE_LIST = pd.read_csv(IMAGE_CSV_PATH, header=None)
PERC_DATA = GoLangJson.PercolationJson(PERCOLATION_JSON_PATH)
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

    # Get percolation data
    # average cluster/box count
    p = PERC_DATA.get_avg_cluster_count_curve(image_name, MIN_R, MAX_R) 
    
    # average biggest cluster area
    g = PERC_DATA.get_avg_biggest_cluster_area_curve(image_name, MIN_R, MAX_R)
    
    # percolation
    h = PERC_DATA.get_percolation_curve(image_name, MIN_R, MAX_R)

    features = [
        lacunarity_curve,
        fractal_dimensions,
        Curve.get_descriptors(R_RANGE, lacunarity_curve),
        p,
        g,
        h,
        Curve.get_descriptors(R_RANGE, p),
        Curve.get_descriptors(R_RANGE, h),
        Curve.get_descriptors(R_RANGE, g),
        [image_class]
    ]
    
    img_features = []
    for feat in features:
        img_features += feat
        
    ARFF_CONTENT.append(img_features)

# 3. Create .arff file
arff = Arff(relation=ARFF_NAME,
            entries=ARFF_CONTENT,
            attrs= PROBABILITY_MATRIX_ARFF_ATTRIBUTES + PERCOLATION_ARFF_ATTRIBUTES, 
            classes=ARFF_CLASSES)
arff.save(ARFF_NAME)

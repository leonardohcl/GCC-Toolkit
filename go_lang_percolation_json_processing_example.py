import pandas as pd
from File import Arff
import GoLangJson
from Curve import Curve
from tqdm import tqdm
from Fractal import PERCOLATION_ARFF_ATTRIBUTES

# 0. Define important variables
# Input details
IMAGE_CSV_PATH = "sample/sample-images.csv"
PERCOLATION_JSON_PATH = "sample/go-percolation-data.json"
MIN_R = 3
MAX_R = 41
R_RANGE = range(MIN_R, MAX_R + 1, 2)


# Arff definitions
ARFF_NAME = "percolation-example"
ARFF_CLASSES = [0, 1]
ARFF_CONTENT = []

# 1. Read image csv and probability matrix
IMAGE_LIST = pd.read_csv(IMAGE_CSV_PATH, header=None)
PERC_DATA = GoLangJson.PercolationJson(PERCOLATION_JSON_PATH)

# 2. Iterate through images to calculate the fractal properties
progress = tqdm(range(len(IMAGE_LIST)))
for idx in progress:
    # Get image name and class
    image_name = IMAGE_LIST.iloc[idx, 0]
    image_class = IMAGE_LIST.iloc[idx, 1]

    progress.set_description(image_name)

    # Get percolation data
    data = PERC_DATA.get_by_name(image_name)
    # If didn't find it, stop the process
    if data.empty:
        raise Exception(
            "Couldn't find the percolation data for {} on the JSON".format(image_name))
    
    # average cluster/box count
    p = PERC_DATA.get_avg_cluster_count_curve(image_name, MIN_R, MAX_R) 
    
    # average biggest cluster area
    g = PERC_DATA.get_avg_biggest_cluster_area_curve(image_name, MIN_R, MAX_R)
    
    # percolation
    h = PERC_DATA.get_percolation_curve(image_name, MIN_R, MAX_R)

    features = p + g + h + Curve.get_descriptors(R_RANGE, p) + Curve.get_descriptors(R_RANGE, h) + Curve.get_descriptors(R_RANGE, g)   [image_class]
    ARFF_CONTENT.append(features)

# 3. Create .arff file
arff = Arff(relation=ARFF_NAME,
            entries=ARFF_CONTENT,
            attrs=PERCOLATION_ARFF_ATTRIBUTES, 
            classes=ARFF_CLASSES)
arff.save(ARFF_NAME)

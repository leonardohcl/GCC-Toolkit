import pandas as pd
from Fractal import GlidingBox
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
ARFF_ATTRIBUTES = ["MinkLAC1",
                   "MinkLAC2",
                   "MinkLAC3",
                   "MinkLAC4",
                   "MinkLAC5",
                   "MinkLAC6",
                   "MinkLAC7",
                   "MinkLAC8",
                   "MinkLAC9",
                   "MinkLAC10",
                   "MinkLAC11",
                   "MinkLAC12",
                   "MinkLAC13",
                   "MinkLAC14",
                   "MinkLAC15",
                   "MinkLAC16",
                   "MinkLAC17",
                   "MinkLAC18",
                   "MinkLAC19",
                   "MinkLAC20",
                   "MinkDF1",
                   "MinkDF2",
                   "MinkDF3",
                   "MinkDF4",
                   "MinkDF5",
                   "MinkDF6",
                   "MinkDF7",
                   "MinkDF8",
                   "MinkDF9",
                   "MinkDF10",
                   "MinkDF11",
                   "MinkDF12",
                   "MinkDF13",
                   "MinkDF14",
                   "MinkDF15",
                   "MinkDF16",
                   "MinkDF17",
                   "MinkDF18",
                   "MinkDF19",
                   "MinkDF20",
                   "MinkAreaLAC",
                   "MinkSkewnessLAC",
                   "MinkAreaRatioLAC",
                   "MinkMaxLAC"]

# 1. Read image csv and probability matrix
IMAGE_LIST = pd.read_csv(IMAGE_CSV_PATH, header=None)
PROB_MATRIX = GoLangJson.GoLangProbabiblityMatrix(PROB_MATRIX_JSON_PATH)

# 2. Iterate through images to calculate the fractal properties
progress = tqdm(range(len(IMAGE_LIST)))
for idx in progress:
    # Get image name and class
    image_name = IMAGE_LIST.iloc[idx, 0]
    image_class = IMAGE_LIST.iloc[idx, 1]

    progress.set_description(image_name)
    
    # Get prabability matrix
    matrix = PROB_MATRIX.getImageMatrix(image_name)
    # If didn't find it, stop the process
    if matrix.empty:
        raise Exception(
            "Couldn't find the probability matrix for {} on the JSON".format(image_name))

    # Create empty lists for the fractal values
    lacunarity_curve = []
    fractal_dimensions = []

    # Process for each r
    for r in R_RANGE:
        # Get probabilities for r
        r_probs = PROB_MATRIX.getProbabilitiesFromImageMatrix(image_name, r)

        # If didn't find it, stop the process
        if r_probs == None:
            raise Exception(
                "Probability matrix didn't have the values for r={} which is between the specified interval {} <= r <= {}".format(r, MIN_R, MAX_R))
        r_lacunarity = GlidingBox.lacunarity([r_probs], r, r)[0]
        r_fractal_dimension = GlidingBox.fractal_dimension([r_probs], r, r)[
            0]
        lacunarity_curve.append(r_lacunarity)
        fractal_dimensions.append(r_fractal_dimension)
    area = Curve.area(R_RANGE, lacunarity_curve)
    skew = Curve.skweness(lacunarity_curve)
    area_ratio = Curve.area_ratio(R_RANGE, lacunarity_curve)
    max_lacunarity = max(lacunarity_curve)
    features = lacunarity_curve + fractal_dimensions + \
        [area, skew, area_ratio, max_lacunarity, image_class]
    ARFF_CONTENT.append(features)

# 3. Create .arff file
arff = Arff(relation=ARFF_NAME,
            entries=ARFF_CONTENT,
            attrs=ARFF_ATTRIBUTES, 
            classes=ARFF_CLASSES)
arff.save(ARFF_NAME)

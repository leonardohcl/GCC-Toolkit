import pandas as pd
from File import Arff
import GoLangJson
from Curve import Curve

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
ARFF_ATTRIBUTES = ["Minkp1",
                   "Minkp2",
                   "Minkp3",
                   "Minkp4",
                   "Minkp5",
                   "Minkp6",
                   "Minkp7",
                   "Minkp8",
                   "Minkp9",
                   "Minkp10",
                   "Minkp11",
                   "Minkp12",
                   "Minkp13",
                   "Minkp14",
                   "Minkp15",
                   "Minkp16",
                   "Minkp17",
                   "Minkp18",
                   "Minkp19",
                   "Minkp20",
                   "Minkg1",
                   "Minkg2",
                   "Minkg3",
                   "Minkg4",
                   "Minkg5",
                   "Minkg6",
                   "Minkg7",
                   "Minkg8",
                   "Minkg9",
                   "Minkg10",
                   "Minkg11",
                   "Minkg12",
                   "Minkg13",
                   "Minkg14",
                   "Minkg15",
                   "Minkg16",
                   "Minkg17",
                   "Minkg18",
                   "Minkg19",
                   "Minkg20",
                   "Minkh1",
                   "Minkh2",
                   "Minkh3",
                   "Minkh4",
                   "Minkh5",
                   "Minkh6",
                   "Minkh7",
                   "Minkh8",
                   "Minkh9",
                   "Minkh10",
                   "Minkh11",
                   "Minkh12",
                   "Minkh13",
                   "Minkh14",
                   "Minkh15",
                   "Minkh16",
                   "Minkh17",
                   "Minkh18",
                   "Minkh19",
                   "Minkh20",
                   "MinkAreaCluster",
                   "MinkSkewnessCluster",
                   "MinkAreaRationCluster",
                   "MinkMaxCluster",
                   "MinkAreaPerc",
                   "MinkSkewnessPerc",
                   "MinkAreaRationPerc",
                   "MinkMaxPerc",
                   "MinkAreaMaxCluster",
                   "MinkSkewnessMaxCluster",
                   "MinkAreaRationMaxCluster",
                   "MinkMaxMaxCluster",
                ]

# 1. Read image csv and probability matrix
IMAGE_LIST = pd.read_csv(IMAGE_CSV_PATH, header=None)
PERC_DATA = GoLangJson.GoLangPercolation(PERCOLATION_JSON_PATH)

# 2. Iterate through images to calculate the fractal properties
for idx in range(len(IMAGE_LIST)):
    # Get image name and class
    image_name = IMAGE_LIST.iloc[idx, 0]
    image_class = IMAGE_LIST.iloc[idx, 1]
    # Get percolation data
    data = PERC_DATA.getPercolationData(image_name)
    # If didn't find it, stop the process
    if data.empty:
        raise Exception(
            "Couldn't find the percolation data for {} on the JSON".format(image_name))

    # Create empty lists for the values
    p = [] # average cluster/box 
    g = [] # average biggest cluster area
    h = [] # percolation

    # Process for each r
    for r in R_RANGE:
        # Get probabilities for r
        perc, avg_cluster_per_box, avg_largest_cluster_size = PERC_DATA.getLocalPercolationData(image_name, r)

        # If didn't find it, stop the process
        if perc == None:
            raise Exception(
                "Percolation data didn't have the values for r={} which is between the specified interval {} <= r <= {}".format(r, MIN_R, MAX_R))
        
        p.append(avg_cluster_per_box)
        g.append(avg_largest_cluster_size)
        h.append(perc)

    p_area = Curve.area(R_RANGE, p)
    p_skew = Curve.skweness(p)
    p_area_ratio = Curve.area_ratio(R_RANGE, p)
    p_max = max(p)
    g_area = Curve.area(R_RANGE, g)
    g_skew = Curve.skweness(g)
    g_area_ratio = Curve.area_ratio(R_RANGE, g)
    g_max = max(g)
    h_area = Curve.area(R_RANGE, h)
    h_skew = Curve.skweness(h)
    h_area_ratio = Curve.area_ratio(R_RANGE, h)
    h_max = max(h)

    features = p + g + h + [p_area, p_skew, p_area_ratio, p_max,
                            h_area, h_skew, h_area_ratio, h_max,
                            g_area, g_skew, g_area_ratio, g_max,
                            image_class]
    ARFF_CONTENT.append(features)

# 3. Create .arff file
arff = Arff(relation=ARFF_NAME,
            entries=ARFF_CONTENT,
            attrs=ARFF_ATTRIBUTES, 
            classes=ARFF_CLASSES)
arff.save(ARFF_NAME)

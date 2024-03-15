import math
import statistics
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook

def get_pixel_coords(x:int, y:int, value: int|tuple) -> list[int]:
        try:
            return [x,y] + list(value)
        except:
            return [x,y, value]
        

def pixelDistanceStr(pixel1, pixel2):
    x1 = pixel1[0]
    y1 = pixel1[1]
    value1 = pixel1[2]
    x2 = pixel2[0]
    y2 = pixel2[1]
    value2 = pixel2[2]
    man = abs(x2-x1) + abs(y2 -y2) + abs(value2-value1)
    euc = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (value2-value1)**2)
    mink = max(abs(x2-x1), abs(y2 -y2), abs(value2-value1))
    return f"man: {man:.2f} | euc: {euc:.2f} | mink: {mink:.2f}"

def pixelStr(pixel):
    return f"({pixel[0]:=2.0f},{pixel[1]:=2.0f})[{pixel[2]:=3.0f}]"

class GlidingBoxN2:
    def _pixel_distance(p1: list, p2: list):
        if (type(p1) != type(p2)):
            raise Exception(
                f"center pixel and reference pixel must be of same type. received {type(p1)} {type(p2)}")

        if (type(p1) == int):
            return abs(p2 - p1)

        channels = len(p1)
        if (channels != len(p2)):
            raise Exception(
                f"center pixel and reference pixel does not have the same dimensions ({channels} against {len(p2)})")
        
        max_diff = -float('inf')
        for i in range(channels):
            diff = abs(p2[i] - p1[i])
            if diff > max_diff: max_diff = diff
        return max_diff
        
    def _pixel_is_in_the_box(center: list, pixel: list, r: int):
        """Checks if a pixel is contained in a box of size r using the chessboard distance. Args:
        center: values for the pixel in the center of the box
        pixel: values for the reference pixel to check if is in the box
        r: size of the square box
        """
        dist = GlidingBoxN2._pixel_distance(center, pixel)
        return dist <= r

    def _label_pixel_cluster(binary_map: list, cluster_map: list, x: int, y: int, label: int, connectivity: int):
        """Recursivelly apply label to pixel cluster
            binary_map: MxN map with zeros and ones to use as reference for clustering
            cluster_map: MxN map with where the labels should be applied
            x: height index of the pixel to be labeled 
            y: width index of the pixel to be labeled 
            label: label to be applied to the pixel cluster
            conn: what kind of neighbor connectivity to check for clusters, should be 4 or 8

        """
        if (cluster_map[x][y] == 0 and binary_map[x][y] != 0):
            [height, width] = np.shape(binary_map)  # get map dimentions
            cluster_map[x][y] = label
            can_look_up = (x - 1) >= 0
            can_look_right = (y + 1) < width
            can_look_left = (y - 1) >= 0
            can_look_down = (x + 1) < height

            if (can_look_up):
                GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map, x -
                                                1, y, label, connectivity)
            if (can_look_left):
                GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map, x,
                                                y - 1, label, connectivity)
            if (can_look_right):
                GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map, x,
                                                y + 1, label, connectivity)
            if (can_look_down):
                GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map, x +
                                                1, y, label, connectivity)

            if (connectivity == 8):
                if (can_look_left):
                    if (can_look_up):
                        GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map,
                                                        x - 1, y - 1, label, connectivity)
                    if (can_look_down):
                        GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map,
                                                        x + 1, y - 1, label, connectivity)
                if (can_look_right):
                    if (can_look_up):
                        GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map,
                                                        x - 1, y + 1, label, connectivity)
                    if (can_look_down):
                        GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map,
                                                        x + 1, y + 1, label, connectivity)

    def _label_clusters(binary_map: list, connectivity: int):
        """Returns map of clusters labeled with numbers and amount of clusters found
            binary_map: MxN map with zeros and ones to find where the clusters of ones are
            connectivity: what kind of neighbor connectivity to check for clusters, should be 4 or 8
        """

        [height, width] = np.shape(binary_map)  # get map dimentions
        current_label = 1  # set starting label for clusters
        cluster_map = np.zeros((height, width))  # create cluster mapping

        # loops through the map
        for x in range(height):
            for y in range(width):
                # if the pixel isn't marked, ignore it
                if (binary_map[x][y] == 0):
                    continue
                # else, if the pixel is marked but it's cluster is not mapped start marking it
                elif (cluster_map[x][y] == 0):
                    GlidingBoxN2._label_pixel_cluster(binary_map, cluster_map, x,
                                                    y, current_label, connectivity)
                    current_label += 1

        return cluster_map, current_label - 1

    def _region_cluster_data(binary_map: list, connectivity: int = 4):
        """Returns the cluster data information
            binary_map: MxN map with zeros and ones to find where the clusters of ones are
            connectivity: what kind of neighbor connectivity to check for clusters, should be 4 or 8
        """
        cluster_map, cluster_count = GlidingBoxN2._label_clusters(binary_map,
                                                                connectivity)
        flat_map = cluster_map.flatten().tolist()

        biggest_cluster_label = statistics.mode(
            filter(lambda x: (x > 0), flat_map))  # gets most frequent label
        # get size in pixels of the cluster with that label
        biggest_cluster_size = flat_map.count(biggest_cluster_label)

        return cluster_map, cluster_count, biggest_cluster_size

    @staticmethod
    def probability_matrix(img_path: str,
                           min_r: int = 3,
                           max_r: int = 11,
                           image_mode: str = "RGB",
                           print_progress=True,
                           is_notebook_env=False):
        """Creates a probability matrix using the gliding box algorithm with chessboard distance.
            Args:
                path: path to the image being processed
                min_r: minimum size of the gliding box. (default 3)
                max_r: maximum size of the gliding box. (default 11)
                image_mode: mode to process input image. Accepts PIL modes, such as "L" for B/W images or "RGB" for RGB images. (default "RGB") 
        """

        # opens file
        file = Image.open(img_path)

        # convert file data to image on given mode
        img = file.convert(image_mode)

        # create empty occurence matrix
        occurrence_matrix = []
        probability_matrix = []

        # get image size
        shape = np.shape(img)
        width = shape[0]
        height = shape[1]

        # iterate over box sizes
        if print_progress:
            if is_notebook_env:
                iterator = tqdm_notebook(range(min_r, max_r + 2, 2))
            else:
                iterator = tqdm(range(min_r, max_r + 2, 2))
        else:
            iterator = range(min_r, max_r + 2, 2)
        for r in iterator:
            if print_progress:
                iterator.set_description(f"[{img_path}] Processing (r={r})")
            # count how many boxes fit in the image
            box_count = (width - r + 1) * (height-r + 1)

            # occurences
            occurrences = np.zeros(pow(r, 2), dtype=int)

            # get pad size from borders
            pad = int(r / 2)

            # get starting and ending indexes for central pixels
            x_start = y_start = pad
            x_end = width - pad
            y_end = height - pad

            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    central_pixel = img.getpixel((y, x))
                    mass = 0
                    # iterate over the box
                    for i in range(x - pad, x + pad + 1):
                        for j in range(y - pad, y + pad + 1):
                            pixel = img.getpixel((j, i))
                            if (GlidingBoxN2._pixel_is_in_the_box(central_pixel, pixel, r)):
                                mass = mass + 1
                    # increment occurence matrix
                    occurrences[mass - 1] = occurrences[mass - 1] + 1

            # add occurrences to matrix
            occurrence_matrix.append(occurrences)
            # calculate probability
            probability_matrix.append(
                [count/box_count for count in occurrences])

        return probability_matrix

    @staticmethod
    def lacunarity(probability_matrix: list, min_r: int = 3, max_r: int = 11) -> float:
        if (max_r < min_r):
            raise Exception("Invalid values for min_r and max_r")
        r_list = [r for r in range(min_r, max_r+1, 2)]
        lac = np.zeros(len(r_list), dtype=float)
        idx = -1
        for r in r_list:
            idx = idx + 1
            max_mass = pow(r, 2)
            first_moment = 0
            second_moment = 0

            for m in range(max_mass):
                first_moment = first_moment + \
                    (m+1) * probability_matrix[idx][m]
                second_moment = second_moment + \
                    pow((m+1), 2) * probability_matrix[idx][m]
            lac[idx] = (second_moment - pow(first_moment, 2)) / \
                pow(first_moment, 2)

        return lac

    @staticmethod
    def fractal_dimension(probability_matrix: list, min_r: int = 3, max_r: int = 11) -> float:
        if (max_r < min_r):
            raise Exception(
                "Invalid values for min_r and max_R".format(min_r, max_r))
        r_list = [r for r in range(min_r, max_r+1, 2)]
        fd = np.zeros(len(r_list), dtype=float)
        idx = -1
        for r in r_list:
            idx = idx + 1
            max_mass = pow(r, 2)
            d = 0

            for m in range(max_mass):
                d = d + probability_matrix[idx][m] / (m+1)
            fd[idx] = d

        return fd

    @staticmethod
    def percolation(img_path: str,
                    min_r: int = 3,
                    max_r: int = 11,
                    percolation_threshold: float = 0.59275,
                    image_mode: str = "RGB",
                    print_progress=True,
                    is_notebook_env=False):
        """Creates a probability matrix using the gliding box algorithm with chessboard distance.
            Args:
                img_path: path to the image being processed
                min_r: minimum size of the gliding box. (default 3)
                max_r: maximum size of the gliding box. (default 11)
                image_mode: mode to process input image. Accepts PIL modes, such as "L" for B/W images or "RGB" for RGB images. (default "RGB") 
        """

        # opens file
        file = Image.open(img_path)

        # convert file data to image on given mode
        img = file.convert(image_mode)

        # get image size
        shape = np.shape(img)
        width = shape[0]
        height = shape[1]
        average_cluster_count = []
        average_biggest_cluster_area = []
        percolation = []

        # iterate over box sizes
        if print_progress:
            if is_notebook_env:
                iterator = tqdm_notebook(range(min_r, max_r + 2, 2))
            else:
                iterator = tqdm(range(min_r, max_r + 2, 2))
        else:
            iterator = range(min_r, max_r + 2, 2)
        for r in iterator:
            if print_progress:
                iterator.set_description(f"[{img_path}] Processing (r={r})")

            # count how many boxes fit in the image
            box_count = (width - r + 1) * (height-r + 1)
            box_area = math.pow(r, 2)

            # get pad size from borders
            pad = int(r / 2)

            # get starting and ending indexes for central pixels
            x_start = y_start = pad
            x_end = width - pad
            y_end = height - pad

            cluster_counts = []
            biggest_cluster_areas = []
            percolation_box_count = 0

            # loop trough central pixels
            for x in range(x_start, x_end):
                for y in range(y_start, y_end):
                    central_pixel = img.getpixel((y, x))
                    region = []
                    percolating_pixels = 0

                    # iterate over the box
                    for i in range(x - pad, x + pad + 1):
                        region_row = []
                        for j in range(y - pad, y + pad + 1):
                            pixel = img.getpixel((j, i))
                            if (GlidingBoxN2._pixel_is_in_the_box(central_pixel, pixel, r)):
                                # if pixel is in the box, tag it on the binary map
                                region_row.append(1)
                                # and count it a as percolated
                                percolating_pixels += 1
                            else:
                                # otherwise leave it as 0
                                region_row.append(0)
                        region.append(region_row)
                    [_,
                     clusters_on_box,
                     biggest_cluster_size] = GlidingBoxN2._region_cluster_data(region)

                    cluster_counts.append(clusters_on_box)
                    biggest_cluster_areas.append(
                        biggest_cluster_size / box_area)
                    if ((percolating_pixels / box_area) >= percolation_threshold):
                        percolation_box_count += 1
            box_average_cluster_count = statistics.mean(
                cluster_counts)
            box_average_biggest_cluster_area = statistics.mean(
                biggest_cluster_areas)
            box_percolation = percolation_box_count/box_count

            average_cluster_count.append(box_average_cluster_count)
            average_biggest_cluster_area.append(
                box_average_biggest_cluster_area)
            percolation.append(box_percolation)

        return PercolationData(min_r,
                               max_r,
                               percolation,
                               average_cluster_count,
                               average_biggest_cluster_area)
class PercolationData:
    def __init__(self, min_r, max_r, percolation, avg_cluster_count, avg_biggest_cluster_area) -> None:
        self._min_r = min_r
        self._max_r = max_r

        # known as g(r) on Guilherme Freire's publications
        self._percolation = percolation

        # known as p(r) on Guilherme Freire's publications
        self._avg_cluster_count = avg_cluster_count

        # known as h(r) on Guilherme Freire's publications
        self._avg_biggest_cluster_area = avg_biggest_cluster_area

    @property
    def min_r(self):
        return self._min_r

    @property
    def max_r(self):
        return self._max_r

    @property
    def percolation(self):
        return self._percolation

    @property
    def avg_cluster_count(self):
        return self._avg_cluster_count

    @property
    def avg_biggest_cluster_area(self):
        return self._avg_biggest_cluster_area

PROBABILITY_MATRIX_ARFF_ATTRIBUTES = [
    "MinkLAC1",
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
    "MinkMaxLAC"
]

PERCOLATION_ARFF_ATTRIBUTES = [
    "Minkp1",
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
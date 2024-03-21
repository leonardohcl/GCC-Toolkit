import math
import statistics
import numpy as np
from PIL import Image
from tqdm import tqdm
from File import ImageMode, ImageFile
from Hyperspace import DistanceMode, Hypercube
from tqdm.notebook import tqdm as tqdm_notebook
from scipy.spatial import KDTree

class GlidingBox:
    @staticmethod
    def get_iterator(iterator, print_progress, is_notebook, leave=bool | None, position=None, ):
        if print_progress:
            if is_notebook:
                return tqdm_notebook(iterator, position=position, leave=leave)
            else:
                return tqdm(iterator, position=position, leave=leave)
        return iterator

    @staticmethod
    def box_weight(coords: list[int],
                   img: ImageFile,
                   r: int,
                   distance=DistanceMode.MINKOWSKI,                   
                   color_tree: KDTree = None,
                   position_tree: KDTree = None):

        pad = int(r/2)
        colors = img.pixels
        hypervector_tree = KDTree(colors) if color_tree == None else color_tree
        image_area_tree = img.area.get_kdTree() if position_tree == None else position_tree 

        center_idx = img.get_pixel_idx(coords)
        center_point = colors[center_idx]

        in_radius = hypervector_tree.query_ball_point(center_point, r= r, p=distance.value)
        in_pad = image_area_tree.query_ball_point(coords, r = pad, p = DistanceMode.MINKOWSKI.value)
        intersection = list(set(in_radius) & set(in_pad))
        
        return len(intersection)

    @staticmethod
    def box_probability(
            img: ImageFile,
            r: int,
            distance=DistanceMode.MINKOWSKI,
            color_tree: KDTree = None,
            position_tree: KDTree = None,
            print_progress=True,
            is_notebook_env=False):

        pad = int(r/2)
        box_count = (img.width - r + 1) * (img.height - r + 1)
        occurrences = {}
        center_pixels = Hypercube([pad, pad], [img.width-pad-pad, img.height-pad-pad])
        
        hypervector_tree = KDTree(img.pixels) if color_tree == None else color_tree
        image_area_tree = img.area.get_kdTree() if position_tree == None else position_tree 

        pixel_iterator = GlidingBox.get_iterator(range(center_pixels.point_count), 
                                                 print_progress,
                                                 is_notebook_env, 
                                                 position=0 if color_tree == None else 1, 
                                                 leave=color_tree == None)
        if print_progress:
            pixel_iterator.set_description(f"r = {r:=2d}")

        def count_weight(point):
            weight = GlidingBox.box_weight(point, img, r, 
                                           distance=distance, 
                                           color_tree=hypervector_tree, 
                                           position_tree=image_area_tree)
            current_count = occurrences.get(weight)
            occurrences[weight] = 1 if current_count == None else current_count + 1

            if print_progress:
                pixel_iterator.update(1)

        center_pixels.loop(count_weight)
        if print_progress:
            pixel_iterator.close()

        if box_count == 0: return {}

        probs = {}
        for weight in occurrences:
            probs[weight] =  occurrences[weight] / box_count
        return probs

    @staticmethod
    def probability_matrix(img_path: str,
                           min_r: int = 3,
                           max_r: int = 5,
                           mode=ImageMode.RGB,
                           distance=DistanceMode.MINKOWSKI,
                           print_progress=True,
                           is_notebook_env=False):
        """Creates a probability matrix using the gliding box algorithm with chessboard distance.
            Args:
                path: path to the image being processed
                min_r: minimum size of the gliding box. (default 3)
                max_r: maximum size of the gliding box. (default 11)
                image_mode: mode to process input image. Accepts PIL modes, such as "L" for B/W images or "RGB" for RGB images. (default "RGB") 
        """

        box_sizes = [r for r in range(min_r, max_r + 1, 2)]

        box_size_iterator = GlidingBox.get_iterator(box_sizes, 
                                                    print_progress, 
                                                    is_notebook_env, 
                                                    position=0)

        if print_progress:
            box_size_iterator.set_description(f"{img_path} | prob. matrix | opening file...")
            box_size_iterator.update(0)
        img = ImageFile(img_path, mode)

        if print_progress:
            box_size_iterator.set_description(f"{img_path} | prob. matrix | building k-d tree...")
        color_tree = KDTree(img.pixels)
        pixel_tree = img.area.get_kdTree()

        probability_matrix = {}
        if print_progress:
            box_size_iterator.set_description(f"{img_path} | prob. matrix | {min_r} <= r <= {max_r}")

        for box in box_size_iterator:
            probability_matrix[box] = GlidingBox.box_probability(img, box,
                                               distance= distance,
                                               position_tree= pixel_tree,
                                               color_tree= color_tree,
                                               print_progress= print_progress,
                                               is_notebook_env= is_notebook_env)

        return probability_matrix

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
        dist = GlidingBox._pixel_distance(center, pixel)
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
                GlidingBox._label_pixel_cluster(binary_map, cluster_map, x -
                                                1, y, label, connectivity)
            if (can_look_left):
                GlidingBox._label_pixel_cluster(binary_map, cluster_map, x,
                                                y - 1, label, connectivity)
            if (can_look_right):
                GlidingBox._label_pixel_cluster(binary_map, cluster_map, x,
                                                y + 1, label, connectivity)
            if (can_look_down):
                GlidingBox._label_pixel_cluster(binary_map, cluster_map, x +
                                                1, y, label, connectivity)

            if (connectivity == 8):
                if (can_look_left):
                    if (can_look_up):
                        GlidingBox._label_pixel_cluster(binary_map, cluster_map,
                                                        x - 1, y - 1, label, connectivity)
                    if (can_look_down):
                        GlidingBox._label_pixel_cluster(binary_map, cluster_map,
                                                        x + 1, y - 1, label, connectivity)
                if (can_look_right):
                    if (can_look_up):
                        GlidingBox._label_pixel_cluster(binary_map, cluster_map,
                                                        x - 1, y + 1, label, connectivity)
                    if (can_look_down):
                        GlidingBox._label_pixel_cluster(binary_map, cluster_map,
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
                    GlidingBox._label_pixel_cluster(binary_map, cluster_map, x,
                                                    y, current_label, connectivity)
                    current_label += 1

        return cluster_map, current_label - 1

    def _region_cluster_data(binary_map: list, connectivity: int = 4):
        """Returns the cluster data information
            binary_map: MxN map with zeros and ones to find where the clusters of ones are
            connectivity: what kind of neighbor connectivity to check for clusters, should be 4 or 8
        """
        cluster_map, cluster_count = GlidingBox._label_clusters(binary_map,
                                                                connectivity)
        flat_map = cluster_map.flatten().tolist()

        biggest_cluster_label = statistics.mode(
            filter(lambda x: (x > 0), flat_map))  # gets most frequent label
        # get size in pixels of the cluster with that label
        biggest_cluster_size = flat_map.count(biggest_cluster_label)

        return cluster_map, cluster_count, biggest_cluster_size

    @staticmethod
    def lacunarity(probability_matrix: dict, min_r: int = 3, max_r: int = 11) -> float:
        if (max_r < min_r): raise Exception("Invalid values for min_r and max_r")
        r_list = [r for r in range(min_r, max_r+1, 2)]
        lac = np.zeros(len(r_list), dtype=float)
        for idx in range(len(r_list)):
            r = r_list[idx]
            max_mass = pow(r, 2)
            first_moment = 0
            second_moment = 0
            r_probabilities = probability_matrix.get(r)
            if r_probabilities == None: 
                lac[idx] = float("inf")
                continue

            for mass_idx in range(max_mass):
                mass = mass_idx + 1
                prob = r_probabilities.get(mass)
                if prob == None: continue

                first_moment += mass * prob
                second_moment += pow(mass, 2) * prob

            lac[idx] = (second_moment - pow(first_moment, 2)) / \
                pow(first_moment, 2)

        return lac

    @staticmethod
    def fractal_dimension(probability_matrix: dict, min_r: int = 3, max_r: int = 11) -> float:
        if (max_r < min_r): raise Exception("Invalid values for min_r and max_R".format(min_r, max_r))

        r_list = [r for r in range(min_r, max_r+1, 2)]
        fd = np.zeros(len(r_list), dtype=float)
        idx = -1
        for idx in range(len(r_list)):
            r = r_list[idx]
            max_mass = pow(r, 2)
            d = 0
            r_probabilities = probability_matrix.get(r)
            if r_probabilities == None: 
                fd[idx] = float("inf")
                continue

            for mass_idx in range(max_mass):
                mass = mass_idx + 1
                prob = r_probabilities.get(mass)
                if prob == None: continue
                d += prob/mass
            fd[idx] = d

        return fd

    @staticmethod
    def percolation(img_path: str,
                    min_r: int = 3,
                    max_r: int = 11,
                    percolation_threshold: float = 0.59275,
                    image_mode = ImageMode.RGB,
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
        img = file.convert(image_mode.value)

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
                            if (GlidingBox._pixel_is_in_the_box(central_pixel, pixel, r)):
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
                     biggest_cluster_size] = GlidingBox._region_cluster_data(region)

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
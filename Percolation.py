import math
import numpy as np
import time
import statistics
from PIL import Image
from GlidingBoxHelper import PixelIsInTheBox


def LabelClusterPixel(binary_map: list, cluster_map: list, x: int, y: int, label: int, connectivity: int):
    """Recursivelly apply label to pixel cluster
        binary_map: MxN map with zeros and ones to use as reference for clustering
        cluster_map: MxN map with where the labels should be applied
        x: height index of the pixel to be labeled 
        y: width index of the pixel to be labeled 
        label: label to be applied to the pixel cluster
        conn: what kind of neighbor connectivity to check for clusters, should be 4 or 8

    """
    if(cluster_map[x][y] == 0 and binary_map[x][y] != 0):
        [height, width] = np.shape(binary_map)  # get map dimentions
        cluster_map[x][y] = label
        can_look_up = (x - 1) >= 0
        can_look_right = (y + 1) < width
        can_look_left = (y - 1) >= 0
        can_look_down = (x + 1) < height

        if(can_look_up):
            LabelClusterPixel(binary_map, cluster_map, x -
                              1, y, label, connectivity)
        if(can_look_left):
            LabelClusterPixel(binary_map, cluster_map, x,
                              y - 1, label, connectivity)
        if(can_look_right):
            LabelClusterPixel(binary_map, cluster_map, x,
                              y + 1, label, connectivity)
        if(can_look_down):
            LabelClusterPixel(binary_map, cluster_map, x +
                              1, y, label, connectivity)

        if(connectivity == 8):
            if(can_look_left):
                if(can_look_up):
                    LabelClusterPixel(binary_map, cluster_map,
                                      x - 1, y - 1, label, connectivity)
                if(can_look_down):
                    LabelClusterPixel(binary_map, cluster_map,
                                      x + 1, y - 1, label, connectivity)
            if(can_look_right):
                if(can_look_up):
                    LabelClusterPixel(binary_map, cluster_map,
                                      x - 1, y + 1, label, connectivity)
                if(can_look_down):
                    LabelClusterPixel(binary_map, cluster_map,
                                      x + 1, y + 1, label, connectivity)


def LabelClusters(binary_map: list, connectivity: int):
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
            if(binary_map[x][y] == 0):
                continue
            # else, if the pixel is marked but it's cluster is not mapped start marking it
            elif(cluster_map[x][y] == 0):
                LabelClusterPixel(binary_map, cluster_map, x,
                                  y, current_label, connectivity)
                current_label += 1

    return [cluster_map, current_label - 1]


def RegionClusterData(binary_map: list, connectivity: int = 4):
    """Returns the cluster data information
        binary_map: MxN map with zeros and ones to find where the clusters of ones are
        connectivity: what kind of neighbor connectivity to check for clusters, should be 4 or 8
    """
    [cluster_map, cluster_count] = LabelClusters(binary_map, connectivity)
    flat_map = cluster_map.flatten().tolist()

    biggest_cluster_label = statistics.mode(filter(lambda x: (x > 0),flat_map)) # gets most frequent label
    biggest_cluster_size = flat_map.count(biggest_cluster_label) # get size in pixels of the cluster with that label

    return [cluster_map, cluster_count, biggest_cluster_size]


def GlidingBoxPercolation(path: str, min_r: int = 3, max_r: int = 11, percolation_threshold: float = 0.59275, image_mode: str = "RGB", print_progress=True):
    """Creates a probability matrix using the gliding box algorithm with chessboard distance.
        Args:
            path: path to the image being processed
            min_r: minimum size of the gliding box. (default 3)
            max_r: maximum size of the gliding box. (default 11)
            image_mode: mode to process input image. Accepts PIL modes, such as "L" for B/W images or "RGB" for RGB images. (default "RGB") 
    """

    # opens file
    file = Image.open(path)

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
    for r in range(min_r, max_r + 2, 2):
        if print_progress:
            print(f"r={r}")
        start = time.time()
        # count how many boxes fit in the image
        box_count = (width - r + 1) * (height-r + 1)
        box_area = math.pow(r,2)

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
                        if(PixelIsInTheBox(central_pixel, pixel, r)):
                            # if pixel is in the box, tag it on the binary map
                            region_row.append(1)
                            # and count it a as percolated
                            percolating_pixels += 1
                        else:
                            # otherwise leave it as 0 
                            region_row.append(0)
                    region.append(region_row)
                [cluster_map, clusters_on_box,
                    biggest_cluster_size] = RegionClusterData(region)


                cluster_counts.append(clusters_on_box)
                biggest_cluster_areas.append(biggest_cluster_size / box_area)
                if((percolating_pixels / box_area )>= percolation_threshold):
                    percolation_box_count += 1
        end = time.time()
        box_average_cluster_count = statistics.mean(cluster_counts) # or p(r)
        box_average_biggest_cluster_area = statistics.mean(biggest_cluster_areas) # or h(r)
        box_percolation = percolation_box_count/box_count # or g(r)

        average_cluster_count.append(box_average_cluster_count)
        average_biggest_cluster_area.append(box_average_biggest_cluster_area)
        percolation.append(box_percolation)

        if print_progress:
            print(f"[p(r)] average cluster/box: {box_average_cluster_count}")
            print(f"[h(r)] average biggest cluster area: {box_average_biggest_cluster_area}")
            print(f"[g(r)] percolation: {box_percolation}")
            print(f"elapsed {end-start:.4f}s")

    return [percolation, average_cluster_count, average_biggest_cluster_area]
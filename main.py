from tqdm import tqdm, tqdm_notebook
import numpy as np
from scipy.spatial import KDTree
from PIL import Image
from Fractal import GlidingBoxN2
from Hyperspace import Hypercube, DistanceMode
from File import ImageFile, ImageMode
import sys


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
            box_size_iterator.set_description(f"{img_path} | opening file...")
            box_size_iterator.update(0)
        img = ImageFile(img_path, mode)

        if print_progress:
            box_size_iterator.set_description(f"{img_path} | building k-d tree...")
        color_tree = KDTree(img.pixels)
        pixel_tree = img.area.get_kdTree()

        probability_matrix = {}
        if print_progress:
            box_size_iterator.set_description(f"{img_path} | {min_r} <= r <= {max_r}")

        for box in box_size_iterator:
            probability_matrix[box] = GlidingBox.box_probability(img, box,
                                               distance= distance,
                                               position_tree= pixel_tree,
                                               color_tree= color_tree,
                                               print_progress= print_progress,
                                               is_notebook_env= is_notebook_env)

        return probability_matrix


PRINT_PROGRESS = True
R = 3
MIN_R = 19
MAX_R = 21
# IMG_PATH = 'asd/test.jpg'
IMG_PATH = 'sample/images/class1.jpg'

for r in range(MIN_R, MAX_R+1, 2):
    print("\n[OLD IMPLEMENTATION]")
    m = GlidingBoxN2.probability_matrix(IMG_PATH, r, r, print_progress=PRINT_PROGRESS)
    # print(m)

    # print("\n[NEW IMPLEMENTATION]")
    # m = GlidingBox.probability_matrix(IMG_PATH, r, r, print_progress=PRINT_PROGRESS)
    # print(m)


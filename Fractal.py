import time
import math
import statistics
import numpy as np
from PIL import Image
from typing import Union


def PixelIsInTheBox(center: Union[list, int], pixel: Union[list, int], r: int):
    """Checks if a pixel is contained in a box of size r using the chessboard distance. Args:
            center: values for the pixel in the center of the box
            pixel: values for the reference pixel to check if is in the box
            r: size of the square box
    """
    if(type(center) != type(pixel)):
        raise Exception(
            f"center pixel and reference pixel must be of same type. received {type(center)} {type(pixel)}")

    if(type(center) == int):
        return abs(pixel - center) <= r

    channels = len(center)
    if (channels != len(pixel)):
        raise Exception(
            f"center pixel and reference pixel does not have the same dimensions ({channels} against {len(pixel)})")
    for i in range(channels):
        diff = abs(pixel[i] - center[i])
        if(diff > r):
            return False
    return True


def GetIntervalSubdivisions(start: int, end: int, blocks: int):
    size = end - start
    block_size = math.ceil(size / blocks)
    intervals = []
    last = start
    for n in range(blocks):
        next = last + block_size
        if next > end:
            next = end
        intervals.append((last, next))
        last = next
    return intervals


def LacunarityFromProabilityMatrix(matrix: list, min_r: int = 3, max_r: int = 11):
    if(max_r < min_r):
        raise Exception(
            "Invalid values for min_r and max_R".format(min_r, max_r))
    r_list = [r for r in range(min_r, max_r+1, 2)]
    lac = np.zeros(len(r_list), dtype=float)
    idx = -1
    for r in r_list:
        idx = idx + 1
        max_mass = pow(r, 2)
        first_moment = 0
        second_moment = 0

        for m in range(max_mass):
            first_moment = first_moment + (m+1) * matrix[idx][m]
            second_moment = second_moment + pow((m+1), 2) * matrix[idx][m]
        lac[idx] = (second_moment - pow(first_moment, 2)) / \
            pow(first_moment, 2)

    return lac


def FractalDimensionFromProabilityMatrix(matrix: list, min_r: int = 3, max_r: int = 11):
    if(max_r < min_r):
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
            d = d + matrix[idx][m] / (m+1)
        fd[idx] = d

    return fd


def GlidingBoxProbabilityMatrix(path: str, min_r: int = 3, max_r: int = 11, image_mode: str = "RGB", print_progress=True):
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

    # create empty occurence matrix
    occurrence_matrix = []
    probability_matrix = []

    # get image size
    shape = np.shape(img)
    width = shape[0]
    height = shape[1]

    # iterate over box sizes
    for r in range(min_r, max_r + 2, 2):
        if print_progress:
            print(f"r={r}")
        start = time.time()
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
                mass = 1
                # iterate over the box
                for i in range(x - pad, x + pad + 1):
                    for j in range(y - pad, y + pad + 1):
                        if (i == x and j == y):
                            continue
                        pixel = img.getpixel((j, i))
                        if(PixelIsInTheBox(central_pixel, pixel, r)):
                            mass = mass + 1
                # increment occurence matrix
                occurrences[mass - 1] = occurrences[mass - 1] + 1

        # add occurrences to matrix
        occurrence_matrix.append(occurrences)
        # calculate probability
        probability_matrix.append([count/box_count for count in occurrences])

        end = time.time()
        if print_progress:
            print(f"elapsed {end-start:.4f}s")

    return probability_matrix


def AreaUnderCurve(x, y):
    if len(x) != len(y):
        raise Exception("X and Y lists don't match length")
    n = len(x)
    a = x[0]
    b = x[n - 1]
    sum = 0
    for idx in range(len(x) - 1):
        sum += y[idx] + y[idx + 1]
    return ((b-a)/(2*n)) * sum


def CurveSkweness(fx):
    n = len(fx)
    fx_mean = statistics.mean(fx)
    top = sum(pow(value - fx_mean, 3) for value in fx) / n
    bottom = math.sqrt(pow(sum(pow(value - fx_mean, 2) for value in fx) / n, 3))
    return top/bottom


def CurveAreaRatio(x, y):
    if len(x) != len(y):
        raise Exception("X and Y lists don't match length")
    split_point = math.floor(len(x) / 2)
    return AreaUnderCurve(x[:split_point], y[:split_point]) / AreaUnderCurve(x[split_point:], y[split_point:])

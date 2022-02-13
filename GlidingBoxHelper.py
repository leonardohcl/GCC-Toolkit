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


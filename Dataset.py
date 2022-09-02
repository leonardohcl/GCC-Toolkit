from copy import deepcopy
import os
import torch
import pandas as pd
import numpy as np
import FileHandling
from torch.utils.data import Dataset


class ImageDatasetEntry:
    def __init__(self, root_dir: str, filename: str, class_id: int, transform: list = None) -> None:
        self._filename = filename
        self._root_dir = root_dir
        self._class_id = class_id
        self._transform = transform

    @property
    def full_path(self):
        return os.path.join(self._root_dir, self._filename)

    @property
    def filename(self):
        return self._filename

    @property
    def class_id(self):
        return self._class_id

    def get_image(self):
        return FileHandling.readImage(self.full_path)

    def get_tensor(self):
        image = self.get_image()
        return self._transform(image) if self._transform else image


class ImageDataset(Dataset):
    """Image dataset"""

    def __init__(self, csv_file: str, root_dir: str, classes: list, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with all images names and classes.
            root_dir (string): Directory with all the images.
            classes(list): List of classes in the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
            loadToGpu (bool): If should load image data to GPU, in case of availability

        Notes:
            Classes are expected to be sequential numeric ints to make possible to emulate the output expected in form of an array.
            Csv files with the listing ust not contain headers and each entry must be filename,class separated by a line break
        """
        image_list = pd.read_csv(csv_file, header=None)
        self._classes = classes
        self._class_count = len(classes)
        self._transform = transform
        self._root_dir = root_dir
        self.images = [ImageDatasetEntry(root_dir, row[0], row[1], self._transform)
                       for _, row in image_list.iterrows()]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ind: int):
        return self.images[ind]

    def get_expected_tensor(self, entry) -> torch.Tensor:
        """Get expected output from a network of image at given index.

        Args:
            ind(ImageDatasetEntry | int): Image entry or image's index
        Output:
            Tensor with size ([class_count]), filled with zeros except to the index that matches the image class
        """
        expected_vector = np.zeros(self._class_count)
        image = entry if type(entry) == ImageDatasetEntry else self[entry]
        class_idx = self._classes.index(image.class_id)
        expected_vector[class_idx] = 1.0
        return torch.tensor(expected_vector, dtype=torch.float)

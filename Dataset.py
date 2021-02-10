import os
import torch
import pandas as pd
import numpy as np
import FileHandling
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    """Image dataset"""

    def __init__(self, csv_file, root_dir, class_count, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with all images names and classes.
            root_dir (string): Directory with all the images.
            class_count(int): Amount of classes in the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.

        Notes:
            Classes are expected to be sequential numeric ints to make possible to emulate the output expected in form of an array.
            Csv files with the listing ust not contain headers and each entry must be filename,class separated by a line break
        """
        self.images = pd.read_csv(csv_file, header=None)
        self.class_count = class_count
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, ind:int):
        if torch.is_tensor(ind):
            ind = ind.tolist()

        img_path = os.path.join(self.root_dir,
                                self.images.iloc[ind, 0])
        image = FileHandling.readImage(img_path)

        if self.transform:
            image = self.transform(image)

        return image

    def getFilename(self, ind:int):
        """Get file name from image at given index.
        Args:
            ind(int): Image index"""
        return self.images.iloc[ind, 0]

    def getClass(self, ind:int):
        """Get class number from image at given index.
        Args:
            ind(int): Image index
        """
        return self.images.iloc[ind, 1]

    def getExpected(self, ind:int):
        """Get expected output from a network of image at given index.

        Args:
            ind(int): Image index
        Output:
            Tensor with size ([class_count]), filled with zeros except to the index that matches the image class
        """
        aux = np.zeros(self.class_count)
        classInd = self.getClass(ind)
        aux[classInd] = 1
        return torch.tensor(aux, dtype=torch.float)

    def getChannelsMeanAndStd(self):
        """Get means and standard deviations for each channel of the entire dataset.
                
        IMPORTANT: 
            Dataset content must be a Tensor ou have a transform that converts it's contet to Tensor type for this to work
        
        Output:
            means(list), std(list): lists with calculated means and standard deviations for each channel in the whole dataset 
        """
        R_means = torch.zeros(len(self))
        G_means = torch.zeros(len(self))
        B_means = torch.zeros(len(self))
        R_stds = torch.zeros(len(self))
        G_stds = torch.zeros(len(self))
        B_stds = torch.zeros(len(self))

        for img in range(len(self)):
            R_means[img] = torch.mean(self[img][0])
            G_means[img] = torch.mean(self[img][1])
            B_means[img] = torch.mean(self[img][2])
            R_stds[img] = torch.mean(self[img][0])
            G_stds[img] = torch.mean(self[img][1])
            B_stds[img] = torch.mean(self[img][2])

        R_mean = float(torch.mean(R_means))
        G_mean = float(torch.mean(G_means))
        B_mean = float(torch.mean(B_means))
        R_std = float(torch.std(R_stds))
        G_std = float(torch.std(G_stds))
        B_std = float(torch.std(B_stds))

        return [R_mean, G_mean, B_mean], [R_std, G_std, B_std]
from numpy import imag
import pandas as pd

class GoLangProbabiblityMatrix:
    """ Go Lang probability matrix JSON"""
    def __init__(self, path):
        """
        Args:
            path(string): path to json file        
        """
        self.file = pd.read_json(path)

    def getImageMatrix(self, image_name):
        return self.file.loc[self.file['name'] == image_name]
    
    def getImageMatrixIndex(self, image_name):
        return self.file.index[self.file['name'] == image_name].tolist()[0]        

    def getProbabilitiesFromImageMatrix(self, image_name, r):
        matrix = self.getImageMatrix(image_name)
        if matrix.empty: raise Exception(
            "Couldn't find the probability matrix for {} on the JSON".format(image_name))
        
        matrix_idx = self.getImageMatrixIndex(image_name)
        for result in matrix.result_kernels[matrix_idx]:
            if result["kernel_size"] == r:
                return result["probabilities"]
        return None


class GoLangPercolation:
    """ Go Lang percolation JSON"""
    def __init__(self, path):
        """
        Args:
            path(string): path to json file        
        """
        self.file = pd.read_json(path)

    def getPercolationData(self, image_name):
        return self.file.loc[self.file['name'] == image_name]
    
    def getIdx(self, image_name):
        return self.file.index[self.file['name'] == image_name].tolist()[0]        

    def getLocalPercolationData(self, image_name, r):
        data = self.getPercolationData(image_name)
        if data.empty: raise Exception(
            "Couldn't find the percolation data for {} on the JSON".format(image_name))
        
        idx = self.getIdx(image_name)
        for result in data.percolation_results[idx]:
            if result["kernel_size"] == r:
                return result["percolation"], result["average_cluster_count"], result["average_biggest_cluster_area"]
        return None
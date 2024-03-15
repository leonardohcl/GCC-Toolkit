import pandas as pd
from Fractal import GlidingBoxN2

class GlidingBoxJson:
    def __init__(self, path:str):
        """
        Args:
            path(string): path to json file        
        """
        self.file = pd.read_json(path)
    
    def get_search_params(self, image_name: str):
        tree = image_name.split('/')
        if len(tree) > 1:
            folder = '/'.join(tree[0:-1])
            filename =  tree[-1]
            return [filename, folder]
        return [image_name, '']
    
    def get_by_name(self, image_name:str) -> pd.DataFrame:
        filename, folder = self.get_search_params(image_name)
        if folder:            
            return self.file.loc[(self.file['name'] == filename) & (self.file['folder'].str.contains(folder))]
        return self.file.loc[self.file['name'] == filename]
    
    def get_idx(self, image_name:str) -> int:
        filename, folder = self.get_search_params(image_name)
        if folder:            
            return self.file.index[(self.file['name'] == filename) & (self.file['folder'].str.contains(folder))].to_list()[0]
        return self.file.index[self.file['name'] == filename].to_list()[0]
    
class ProbabilityMatrixJson(GlidingBoxJson):
    """ Go Lang probability matrix JSON"""  

    def get_probabilities(self, image_name, r):
        matrix = self.get_by_name(image_name)
        if matrix.empty: raise Exception(
            "Couldn't find the probability matrix for {} on the JSON".format(image_name))
        
        matrix_idx = self.get_idx(image_name)
        for result in matrix.result_kernels[matrix_idx]:
            if result["kernel_size"] == r:
                return result["probabilities"]
        return None
    
    def get_lacunarity(self, image_name, r) -> float:
        r_probs = self.get_probabilities(image_name, r)
        if r_probs == None: raise Exception("Probability matrix didn't have the values for r={}".format(r))
        return GlidingBoxN2.lacunarity([r_probs], r, r)[0]

    def get_fractal_dimension(self, image_name, r) -> float:
        r_probs = self.get_probabilities(image_name, r)
        if r_probs == None: raise Exception("Probability matrix didn't have the values for r={}".format(r))
        return GlidingBoxN2.fractal_dimension([r_probs], r, r)[0]
    
    def get_lacunarity_curve(self, image_name:str, min_r:int, max_r: int) -> list[float]:
        lac = []
        for r in range(min_r, max_r + 1, 2):
            lac.append(self.get_lacunarity(image_name, r))
        return lac
    
    def get_fractal_dimension_curve(self, image_name:str, min_r:int, max_r: int) -> list[float]:
        df = []
        for r in range(min_r, max_r + 1, 2):
            df.append(self.get_fractal_dimension(image_name, r))
        return df

class PercolationJson(GlidingBoxJson):
    """ Go Lang percolation JSON"""
    def __init__(self, path):
        """
        Args:
            path(string): path to json file        
        """
        self.file = pd.read_json(path)

    def get_attribute(self, image_name: str, r: int, attr: str) -> float:
        data = self.get_by_name(image_name)
        if data.empty: raise Exception(
            "Couldn't find the percolation data for {} on the JSON".format(image_name))        
        idx = self.get_idx(image_name)
        for result in data.percolation_results[idx]:
            if result["kernel_size"] == r:
                return result[attr]
        return None
                
    def get_percolation(self, image_name:str, r:int) -> float:
       return self.get_attribute(image_name, r, "percolation")
   
    def get_avg_cluster_count(self, image_name:str, r:int) -> float:
       return self.get_attribute(image_name, r, "average_cluster_count")
   
    def get_avg_biggest_cluster_area(self, image_name:str, r:int) -> float:
       return self.get_attribute(image_name, r, "average_biggest_cluster_area")
   
    def get_percolation_curve(self, image_name: str, min_r: int, max_r: int):
        h = []
        for r in range(min_r, max_r + 1, 2):
            h.append(self.get_percolation(image_name, r))
        return h
    
    def get_avg_cluster_count_curve(self, image_name: str, min_r: int, max_r: int):
        p = []
        for r in range(min_r, max_r + 1, 2):
            p.append(self.get_avg_cluster_count(image_name, r))
        return p
    
    def get_avg_biggest_cluster_area_curve(self, image_name: str, min_r: int, max_r: int):
        g = []
        for r in range(min_r, max_r + 1, 2):
            g.append(self.get_avg_cluster_count(image_name, r))
        return g
     
    def get_percolation_data(self, image_name, r):
        return [
            self.get_percolation(image_name, r),
            self.get_avg_cluster_count(image_name, r),
            self.get_avg_biggest_cluster_area(image_name, r),
        ]
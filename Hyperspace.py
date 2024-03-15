
import math
from typing import Callable
from enum import Enum
from scipy.spatial import KDTree

class DistanceMode(Enum):
    MANHATTAN = 1
    EUCLIDIAN = 2
    MINKOWSKI = float('inf')
    

class Vector:
    def __init__(self, coords: list[int]):
        self.coords = coords

    @property
    def dimensions(self): return len(self.coords)    
    
    @staticmethod
    def manhattan_distance(coords1: list, coords2: list) -> float:
        dimensions = len(coords1)
        diffs = [abs(coords1[idx] - coords2[idx]) for idx in range(dimensions)]
        return sum(diffs)
    
    @staticmethod
    def euclidian_distance(coords1: list, coords2: list) -> float:
        dimensions = len(coords1)
        diffs = [(coords1[idx] - coords2[idx])**2 for idx in range(dimensions)]
        return math.sqrt(sum(diffs))
    
    @staticmethod
    def minkowski_distance(coords1: list, coords2: list) -> float:
        dimensions = len(coords1)
        diffs = [abs(coords1[idx] - coords2[idx]) for idx in range(dimensions)]
        return max(diffs)

    @staticmethod
    def distance(coords1:list, coords2:list, mode = DistanceMode.EUCLIDIAN) -> float:
        if mode == DistanceMode.EUCLIDIAN: return Vector.euclidian_distance(coords1, coords2)            
        if mode == DistanceMode.MINKOWSKI: return Vector.minkowski_distance(coords1, coords2)
        if mode == DistanceMode.MANHATTAN: return Vector.manhattan_distance(coords1, coords2)
        return float('inf')

class Hypercube:
    def __init__(self, start: list, sizes: list):
        if len(start) > len(sizes): raise f"Missmathcing dimensions for starting point and sizes ({len(start)} and {len(sizes)})"
        self.start = start
        self.sizes = sizes

    @property
    def dimensions(self): return len(self.start)

    @property
    def end(self): return [self.get_dimension_reach(idx) for idx in range(self.dimensions)]

    @property
    def point_count(self):
        mult = 1
        for size in self.sizes:
            mult *= size
        return mult

    def contains(self, point:list[int]):
        for idx in range(self.dimensions):
            contained = point[idx] >= self.start[idx] and point[idx] < self.end[idx]
            if not contained: return False
        return True

    def get_dimension_reach(self, idx): return self.start[idx] + self.sizes[idx]

    def loop(self, fn:Callable, start_point:list[int] = None):
        ref = start_point
        if ref == None: ref = []
        depth = len(ref)

        if depth == self.dimensions:
            return fn(ref)
        
        for coord in range(self.start[depth], self.end[depth]):
            self.loop(fn, ref + [coord])
        
    def get_points(self):
        points = []
        self.loop(lambda point: points.append(point))
        return points
    
    def get_point_idx(self, point:list[int]):
        point_idx = 0
        multiplier = 1
        
        for idx in range(self.dimensions - 1, -1, -1):
            offset = point[idx] - self.start[idx]
            point_idx += offset * multiplier
            multiplier *= self.sizes[idx]
        
        return point_idx 

    def get_kdTree(self):
        return KDTree(self.get_points())
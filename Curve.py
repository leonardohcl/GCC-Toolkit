
import math
import statistics

class Curve: 
    @staticmethod
    def area(x, y):
        if len(x) != len(y):
            raise Exception("X and Y lists don't match length")
        n = len(x)
        a = x[0]
        b = x[n - 1]
        sum = 0
        for idx in range(len(x) - 1):
            sum += y[idx] + y[idx + 1]
        return ((b-a)/(2*n)) * sum

    @staticmethod
    def skweness(fx):
        try:
            n = len(fx)
            fx_mean = statistics.mean(fx)
            top = sum(pow(value - fx_mean, 3) for value in fx) / n
            bottom = math.sqrt(pow(sum(pow(value - fx_mean, 2) for value in fx) / n, 3))
            return top/bottom    
        except:
            return 0.0
        
    @staticmethod    
    def area_ratio(x, y):
        if len(x) != len(y):
            raise Exception("X and Y lists don't match length")
        split_point = math.floor(len(x) / 2)
        return Curve.area(x[:split_point], y[:split_point]) / Curve.area(x[split_point:], y[split_point:])

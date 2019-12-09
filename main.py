from mylib.som import SOM
from mylib.data import Data
from mylib.kdtree import KdTree

import numpy as np


# Start from here!
if(__name__ == "__main__"):
    # Get Data & Data Preprocessing
    data = Data()
    X, y = data.clean()
    # Training SOM model & get Result
    som = SOM(X, y)
    som.train(width=100, height=100)
    new_data = som.get_map()
    # Apply KdTree
    kdtree = KdTree()
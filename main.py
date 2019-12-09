#%%
from mylib.som import SOM
from mylib.data import Data
from mylib.kdtree import KdTree
#%%
import numpy as np
#%%
data = Data()
X, y = data.clean()

#%%
som = SOM(X, y)
som.train(width=100, height=100, epochs=1e4)

# %%
som.show(count=1000)

# %%

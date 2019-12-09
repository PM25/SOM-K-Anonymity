import numpy as np
from minisom import MiniSom
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, features, target):
        self.features = features
        self.nfeatures = features.shape[1]
        self.nrow = features.shape[0]
        self.target = target
        self.markers = ['x', 'o', 'D', '*', '1', 'v', '.', 's']
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    #  Training SOM model
    def train(self, width=10, height=10, sigma=.9, lr=.2, epochs=1e5):
        self.som = MiniSom(width, height, self.nfeatures, sigma=sigma, learning_rate=lr)
        self.som.train_random(self.features, int(epochs), verbose=True)
    
    # Get the SOM results
    def get_map(self, verbose=True, log=1000):
        out = []
        for step, (X, y) in enumerate(zip(self.features, self.target)):
            new_X = self.som.winner(X)
            out.append((new_X, X, y))
            if(verbose == True and step % log == 0):
                print(f'*Creating SOM: [{step}/{self.nrow}]')
        return np.array(out)

    # Plot the SOM
    def show(self, verbose=True, log=1000):
        for step, (new_X, _, y) in enumerate(self.get_map(verbose, log)):
            plt.plot(new_X[0], new_X[1], self.markers[int(y)], color=self.colors[int(y)])
            if(verbose == True and step % log == 0):
                print(f'*Plotting SOM: [{step}/{self  .nrow}]')
        plt.show()
        
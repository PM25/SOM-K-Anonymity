from minisom import MiniSom
import matplotlib.pyplot as plt


class SOM:
    def __init__(self, data, target):
        self.data = data
        self.nfeatures = data.shape[1]
        self.nrow = data.shape[0]
        self.target = target
        self.markers = ['x', 'o', 'D', '*', '1', 'v', '.', 's']
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

    #  Training SOM model
    def train(self, width=10, height=10, sigma=3, lr=.5, neighborhood_func='triangle', epochs=1e5):
        self.som = MiniSom(width, height, self.nfeatures, sigma=sigma, learning_rate=lr, neighborhood_function=neighborhood_func)
        self.som.train_random(self.data, int(epochs), verbose=True)
    
    # Get the SOM results
    def get_map(self, verbose=True, interval=1000):
        out = []
        for step, X in enumerate(self.data):
            new_pos = self.som.winner(X)
            out.append((new_pos, X))
            if(verbose == True and step % interval == 0):
                print(f'*Creating SOM: [{step}/{self.nrow}]')
        return out

    # Plot the SOM
    def show(self, count=1000, verbose=True, log=1000):
        for step, ((som_pos, _), y) in enumerate(zip(self.get_map(verbose, log), self.target)):
            plt.plot(som_pos[0], som_pos[1], self.markers[int(y)], color=self.colors[int(y)])
            if(verbose == True and step % log == 0):
                print(f'*Plotting SOM: [{step}/{self  .nrow}]')
            if(step >= count): break
        plt.show()
        
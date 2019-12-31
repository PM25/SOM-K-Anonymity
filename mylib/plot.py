import matplotlib.pyplot as plt

class Plot:
    def __init__(self, k_anonymous_his, som_k_anonymous_his):
        self.k_anonymous_his = k_anonymous_his
        self.som_k_anonymous_his = som_k_anonymous_his
        self.show_single_measurement('Acc')
        self.show_single_measurement('AUC')
        self.show_single_measurement('Time')
    
    def show_single_measurement(self, key):
        plt.plot(self.k_anonymous_his['K-size'], self.k_anonymous_his[key], label=f'K-Anonymous {key}')
        plt.plot(self.som_k_anonymous_his['K-size'], self.som_k_anonymous_his[key], label=f'SOM K-Anonymous {key}')
        plt.legend()
        plt.show()
import numpy as np

class Func:
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def deriv_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def mse_loss(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()
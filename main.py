import sys
import numpy as np


sys.path.append('src')

from src import Network
from src import dataGen as dg

def main():
    network = Network.Network()
    network.train(dg.X, dg.y)

    emily = np.array([-7, -3]) 
    frank = np.array([20, 2])  
    print(network.feedforward(emily))
    print(network.feedforward(frank)) 

if __name__ == '__main__':
    main()
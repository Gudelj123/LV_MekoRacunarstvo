import sys
import os
import random
import math
import networkx as nx
from sklearn.neural_network import MLPClassifier, MLPRegressor
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import pathlib

N_PER_LAYER = [(5,1), (10,1), (30,1), (5,2), (10,2), (30,2), (5,3), (10,3), (30,3)]
ALPHA = 0.00001
ACTIVATION_F = ['identity','logistic', 'tanh', 'relu']
SOLVER = ['lbfgs', 'sgd', 'adam']
MAX_ITER = 3000

learn_data_10 = np.genfromtxt("func_10.csv", delimiter=',', dtype='float32')
learn_data_30 = np.genfromtxt("func_30.csv", delimiter=',', dtype='float32')
learn_data_60 = np.genfromtxt("func_60.csv", delimiter=',', dtype='float32')
learn_data = [learn_data_10, learn_data_30, learn_data_60]
test_data = np.genfromtxt("func_75.csv", delimiter=',', dtype='float32')

outputs = []
for n in N_PER_LAYER:
    for afn in ACTIVATION_F:
        for solv in SOLVER:
            for i in range(3):
                mlp = MLPRegressor(hidden_layer_sizes=n, activation=afn, solver=solv, alpha=ALPHA, max_iter=MAX_ITER)
                mlp.fit(learn_data[i][:,0].reshape(learn_data[i].shape[0],1), learn_data[i][:,1])
                output = mlp.predict(test_data[:,0].reshape(test_data.shape[0],1))
                np.savetxt('data/' + str(n) + afn + solv + str(i) + '.csv', output, delimiter=',')
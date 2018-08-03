# -*- coding: utf-8 -*-

import numpy as np

plot_unit_width = 3
plot_unit_height = 2
plot_time_unit = 50
plot_cell_size = 0.05

def get_ed(a, b): # Euclidean Distance
    return np.linalg.norm(a-b)

def get_mse(a, b, ax=None): # Mean Squared Error
    return ((a-b)**2).mean(axis=ax)

def get_rmse(a, b, ax=None): # Root Mean Squared Error
    return np.sqrt(get_mse(a, b, ax))

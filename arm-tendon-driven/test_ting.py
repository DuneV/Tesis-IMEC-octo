import numpy as np
from octopustesis import *

parametros = [
    [0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1],
    [0, np.radians(40), np.radians(20), 1, 1, 0.5, 1]
]

h = creation_octopus(parametros, 1)
print(h)
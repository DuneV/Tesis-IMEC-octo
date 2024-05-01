import numpy as np
from octopustesis import *

parametros = [
    [0,  np.radians(20), np.radians(30), 22.755, 22.755, 27, 0],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 1],
    [0,  np.radians(20), np.radians(30), 22.755, 22.755, 27, 2],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 3],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 4],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 5],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 6],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 7],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 8],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 9],
    [0, np.radians(40), np.radians(20), 22.755, 22.755, 27, 10]
] # q_1 (radians), q_2 (radians), q_3 (radians), L1 (mm), L2 (mm), R (mm), index (i)

# parameter 4 and 5 are the total distance
creation_octopus(parametros, 0)
# k = 0.8258/1.5 # N/mm
# springs_n = 3
# constants = np.full((len(parametros), springs_n), k)
# len_in = 45.51
# initial_len = np.full((len(parametros),springs_n), len_in)
# forces = (h - initial_len) * constants
# force_1 = np.sum(forces[:,0])
# force_2 = np.sum(forces[:,1])
# force_3 = np.sum(forces[:,2])
# print( force_1 ) # N
# print( force_2 ) # N
# print( force_3 ) # N
# motor_tor = 1 # Nm
# eje = 25 / 1000 # m
# real_force = motor_tor / eje
# print(real_force)
# # motor 25mm centro
# # def cuasiestatic_analisis():
    
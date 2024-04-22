from sympy.physics.mechanics import ReferenceFrame, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# our libraries

from armSection import Section
from utility_t import *


parametros = [
    (0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0),
    (0, np.radians(-20), np.radians(-30), 1, 1, 0.5, 1)
]

def creation_octopus(parameters):
    N_inicial = ReferenceFrame('N')
    O_inicial = Point('O')
    counter_init = 0
    seccionn = [None] * len(parameters)
    centroid = [0] * (len(parameters)- 1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    O = [0, 0, 0]
    ax.scatter(*O, color='k', s=50, label='O')
    comps = np.full((len(parameters)*3, 3), None, dtype=object) 
    for i in range(len(parameters)):
        if counter_init == 0:
            seccionn[i] = Section(N_inicial, O_inicial, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5],parameters[0][6])
            counter_init += 1 
        else:
            seccionn[i] = Section(seccionn[i-1].B, seccionn[i-1].B_point, parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3], parameters[i][4], parameters[i][5], parameters[i][6])
            centroid[i - 1] = vector_to_components(seccionn[i].coords_B, seccionn[i].B, seccionn[i].values)
        
        seccionn[i].update_values(seccionn[i].values)
        comps[(i * 3) ][:] = vector_to_components(seccionn[i].vector1, seccionn[i].N, seccionn[i].values)
        comps[(i * 3) + 1][:] = vector_to_components(seccionn[i].vector2, seccionn[i].N, seccionn[i].values)
        comps[(i * 3) + 2][:] = vector_to_components(seccionn[i].vector3, seccionn[i].N, seccionn[i].values)
        vectors = {
            f'vector1_{i}': (seccionn[i].vector1, 'b'),
            f'vector2_{i}': (seccionn[i].vector2, 'y'),
            f'vector3_{i}': (seccionn[i].vector3, 'c'),
            f'vector_r1_{i}': (seccionn[i].vector_r1, 'm'),
            f'vector_r2_{i}': (seccionn[i].vector_r2, 'orange'),
            f'vector_r3_{i}': (seccionn[i].vector_r3, 'purple'),
            f'vector_N_{i}': (seccionn[i].vector_N, 'magenta'),
            f'vector_OA_{i}': (seccionn[i].vector_OA,'r'), 
            f'vector_AB_{i}': (seccionn[i].vector_AB, 'g')
        }

        for label, (vector, color) in vectors.items():
            if label == f'vector_N_{i}':
                vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                plot_vector(ax, O, vector_components, color, label)
                C_NC = vector_components
            elif label == 'vector_OA':
                vector_components = vector_to_components(vector, seccion.N , seccion.values)
                plot_vector(ax, O, vector_components, color, label)
                A = vector_components
            elif label == 'vector_AB':
                vector_components = vector_to_components(vector, seccion.N , seccion.values)
                plot_vector(ax, A, vector_components, color, label)
            else:
                vector_components = vector_to_components(vector, seccion.N , seccion.values)
                plot_vector(ax, O, vector_components, color, label)

creation_octopus(parametros)
from sympy.physics.mechanics import ReferenceFrame, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# our libraries

from armSection import Section
from utility_t import *

AXIS_LIM = 4


def creation_octopus(parameters):
    N_inicial = ReferenceFrame('N')
    O_inicial = Point('O')
    counter_init = 0
    seccionn = [None] * len(parameters)
    centroid = np.zeros((len(parameters) - 1 , 3))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    O = [0, 0, 0]
    C_NC = np.zeros((len(parameters), 3))
    A = np.zeros((len(parameters), 3))
    ax.scatter(*O, color='k', s=50, label='O')
    comps = np.full((len(parameters)*3, 3), None, dtype=object) 
    points_center = np.empty((100, 3, 7), dtype=np.float64)
    for i in range(len(parameters)):
        if counter_init == 0:
            seccionn[i] = Section(N_inicial, O_inicial, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5],parameters[0][6])
            counter_init += 1 
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
                    C_NC[i, :] = vector_components
                elif label == f'vector_OA_{i}':
                    vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                    plot_vector(ax, O, vector_components, color, label)
                    A[i, :] = vector_components
                elif label == f'vector_AB_{i}':
                    vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                    plot_vector(ax, A[i, :], vector_components, color, label)
                else:
                    vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                    plot_vector(ax, O, vector_components, color, label)
            comps[(i * 3) ][:] = [comps[(i * 3) ][0] - C_NC[i, 0], comps[(i * 3) ][1] - C_NC[i, 1], comps[(i * 3) ][2] - C_NC[i, 2]]
            comps[(i * 3) + 1][:] = [comps[(i * 3) + 1][0] - C_NC[i, 0], comps[(i * 3) + 1][1] - C_NC[i, 1], comps[(i * 3) + 1][2] - C_NC[i, 2]]
            comps[(i * 3) + 2][:] = [comps[(i * 3) + 2][0] - C_NC[i, 0], comps[(i * 3) + 2][1] - C_NC[i, 1], comps[(i * 3) + 2][2] - C_NC[i, 2]]

            plot_vector(ax, C_NC[i, :], comps[(i * 3) ][:], 'magenta', f'vector_r12_{i}')
            plot_vector(ax, C_NC[i, :], comps[(i * 3) + 1][:], 'g', f'vector_r22_{i}')
            plot_vector(ax, C_NC[i, :], comps[(i * 3) + 2][:], 'r', f'vector_r32_{i}')

            x, y, z = circle_points([0, 0, 0], 0.5, 0)
            points = np.vstack((x, y, z)).T
            angle_rad_init = np.radians(90)
            rotated_points = rotate_x(points, angle_rad_init)
            ax.plot(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], color='black')
            
            points_center[:,:,i] = np.copy(rotated_points)
            center = np.array([C_NC[i,0], C_NC[i, 1], C_NC[i, 2]])
            points_center[:,0,i] +=  center[0]
            points_center[:,1,i] +=  center[1]
            points_center[:,2,i] +=  center[2]

            angle_rad = parameters[0][2]  
            angle_rad2 = parameters[0][1]
            points_center[:,:,i] = rotate_xy(points_center[:,:,i], center, angle_rad)
            points_center[:,:,i] = rotate_yz(points_center[:,:,i], center, angle_rad2)
            ax.plot(points_center[:,0,i], points_center[:,1,i], points_center[:,2,i], color='black')
        else:
            seccionn[i] = Section(seccionn[i-1].B, seccionn[i-1].B_point, parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3], parameters[i][4], parameters[i][5], parameters[i][6])
            seccionn[i].update_values(seccionn[i].values)
            centroid[i - 1][:] = vector_to_components(seccionn[i].coords_B, seccionn[i-1].B, seccionn[i].values)
            comps[(i * 3) ][:] = vector_to_components(seccionn[i].vector1, seccionn[i].N, seccionn[i].values)
            comps[(i * 3) + 1][:] = vector_to_components(seccionn[i].vector2, seccionn[i].N, seccionn[i].values)
            comps[(i * 3) + 2][:] = vector_to_components(seccionn[i].vector3, seccionn[i].N, seccionn[i].values)
            vectors2 = {
                f'vector_N_{i}': (seccionn[i].vector_N, 'magenta'),
                f'vector_OA_{i}': (seccionn[i].vector_OA,'purple'), 
                f'vector_AB_{i}': (seccionn[i].vector_AB, 'g'),
                f'vector3_{i}': (seccionn[i].vector3, 'c'),
                f'vector2_{i}': (seccionn[i].vector2, 'y'),
                f'vector1_{i}': (seccionn[i].vector1, 'b')
                }
            for label, (vector, color) in vectors2.items():
                if label == f'vector_N_{i}':
                    vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                    plot_vector(ax, C_NC[i-1,:], vector_components, color, label)
                    C_NC[i, :] = [a + b for a, b in zip(vector_components, C_NC[i-1,:])]
                    ax.scatter(*C_NC[i], color='g', s=50, label=f'B_{i}')

                elif label == f'vector_OA_{i}':
                    vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                    plot_vector(ax, C_NC[i-1, :], vector_components, color, label)
                    A[i,:] = [a + b for a, b in zip(vector_components, C_NC[i-1,:])]
                elif label == f'vector_AB_{i}':
                    vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                    plot_vector(ax, A[i,:], vector_components, color, label)
                else:
                    vector_components = vector_to_components(vector, seccionn[i].N , seccionn[i].values)
                    plot_vector(ax, C_NC[i-1, :], vector_components, color, label)
            comps[(i * 3) ][:] = [comps[(i * 3) ][0] + C_NC[i - 1, 0] - C_NC[i, 0], comps[(i * 3) ][1] + C_NC[i - 1, 1] - C_NC[i, 1], comps[(i * 3) ][2] + C_NC[i - 1, 2]  - C_NC[i, 2]]
            comps[(i * 3) + 1][:] = [comps[(i * 3) + 1][0] + C_NC[i - 1, 0] - C_NC[i, 0], comps[(i * 3) + 1][1] + C_NC[i - 1, 1]  - C_NC[i, 1], comps[(i * 3) + 1][2] + C_NC[i - 1, 2]  - C_NC[i, 2]]
            comps[(i * 3) + 2][:] = [comps[(i * 3) + 2][0] + C_NC[i - 1, 0] - C_NC[i, 0], comps[(i * 3) + 2][1] + C_NC[i - 1, 1]  - C_NC[i, 1], comps[(i * 3) + 2][2] + C_NC[i - 1, 2]  - C_NC[i, 2]]

            plot_vector(ax, C_NC[i, :], comps[(i * 3) ][:], 'magenta', f'vector_r12_{i}')
            plot_vector(ax, C_NC[i, :], comps[(i * 3) + 1][:], 'g', f'vector_r22_{i}')
            plot_vector(ax, C_NC[i, :], comps[(i * 3) + 2][:], 'r', f'vector_r32_{i}')
            
            if 1 * np.sign(parameters[i][2])*np.sign(angle_rad) > 0:
                angle_rad = parameters[i][2]
            else:
                if parameters[i][2] > 0:
                    angle_rad = -1 * angle_rad + parameters[i][2]
                else:
                    angle_rad = -1 * angle_rad + parameters[i][2]
            


            if 1 * np.sign(parameters[i][1])*np.sign(angle_rad2) > 0:
                angle_rad2 = parameters[i][1]
            else:
                if parameters[i][1] > 0:
                    angle_rad2 = -1 *angle_rad2 + parameters[i][1]
                else:
                    angle_rad2 = -1 * angle_rad2 + parameters[i][1]
            print(angle_rad)
            print(angle_rad2)
            print(parameters[i][2])
            print(parameters[i][1])


            points_center[:,:,i] = np.copy(points_center[:,:, i-1])
            center2 = np.array([C_NC[i, 0], C_NC[i, 1], C_NC[i, 2]])
            points_center[:,0,i] +=  centroid[i - 1, 0]
            points_center[:,1,i] +=  centroid[i - 1, 1]
            points_center[:,2,i] +=  centroid[i - 1, 2]
            points_center[:,:,i] = rotate_xy(points_center[:,:,i], center2, angle_rad)
            points_center[:,:,i] = rotate_yz(points_center[:,:,i], center2, angle_rad2)
            ax.plot(points_center[:,0,i], points_center[:,1,i], points_center[:,2,i], color='black')

    # Configuración del gráfico
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.legend()

    ax.set_xlim(-AXIS_LIM, AXIS_LIM)
    ax.set_ylim(-AXIS_LIM, AXIS_LIM)
    ax.set_zlim(-AXIS_LIM, AXIS_LIM)

    plt.show()

parametros = [
    [0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0],
    [0, np.radians(-20), np.radians(-30), 1, 1, 0.5, 1],
    [0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0]
]
creation_octopus(parametros)
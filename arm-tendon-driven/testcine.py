# external libraries 

from sympy.physics.mechanics import ReferenceFrame, Point
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

# our libraries

from armSection import Section
from utility_t import *

# constanst

AXIS_LIM = 4

# define the initial conditions and the configuration of the angles

N_inicial = ReferenceFrame('N')
O_inicial = Point('O')
parametros = [
    (0, np.radians(20), np.radians(30), 1, 1, 0.5)  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
]

# creation of the Sections

seccion = Section(N_inicial, O_inicial, 0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0)
seccion2 = Section(seccion.B, seccion.B_point, 0, np.radians(-20), np.radians(-30), 1, 1, 0.5, 1)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Actualizar los valores de los vectores en la sección

seccion.update_values(seccion.values)
seccion2.update_values(seccion2.values)

# Graficar el punto O (Origen)
O = [0, 0, 0]
ax.scatter(*O, color='k', s=50, label='O')
B_point = vector_to_components(seccion.coords_B, seccion.N ,seccion.values)

ax.scatter(*B_point, color='k', s=50, label='B')
ref_pointB = B_point

comp2 = vector_to_components(seccion.vector1, seccion.N, seccion.values)
comp3 = vector_to_components(seccion.vector2, seccion.N, seccion.values)
comp4 = vector_to_components(seccion.vector3, seccion.N, seccion.values)

comp = vector_to_components(seccion2.coords_B, seccion.B, seccion2.values)

# Graficar otros vectores desde el origen para simplificar
vectors = {
    'vector1': (seccion.vector1, 'b'),
    'vector2': (seccion.vector2, 'y'),
    'vector3': (seccion.vector3, 'c'),
    'vector_r1': (seccion.vector_r1, 'm'),
    'vector_r2': (seccion.vector_r2, 'orange'),
    'vector_r3': (seccion.vector_r3, 'purple'),
    'vector_N': (seccion.vector_N, 'magenta'),
    'vector_OA': (seccion.vector_OA,'r'), 
    'vector_AB': (seccion.vector_AB, 'g')
}

for label, (vector, color) in vectors.items():
    if label == 'vector_N':
        vector_components = vector_to_components(vector, seccion.N , seccion.values)
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

comp1 = [x + y for x, y in zip(comp, C_NC)]
ax.scatter(*comp1, color='g', s=50, label='B2')

plot_vector(ax, C_NC, comp, 'red', 'vector_N2' )

comp2 = [comp2[0] - B_point[0], comp2[1] - B_point[1], comp2[2] - B_point[2]]
comp3 = [comp3[0] - B_point[0], comp3[1] - B_point[1], comp3[2] - B_point[2]]
comp4 = [comp4[0] - B_point[0], comp4[1] - B_point[1], comp4[2] - B_point[2]]

plot_vector(ax, B_point, comp2, 'magenta', 'vector_r12')
plot_vector(ax, B_point, comp3, 'g', 'vector_r22')
plot_vector(ax, B_point, comp4, 'r', 'vector_r32')

# circulo 1

x, y, z = circle_points([0, 0, 0], 0.5, 0)
points = np.vstack((x, y, z)).T
angle_rad = np.radians(90)
rotated_points = rotate_x(points, angle_rad)

ax.plot(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], color='black')

# circulo 2

points2 = np.copy(rotated_points)
center = np.array([C_NC[0], C_NC[1], C_NC[2]])
points2[:, 0] += center[0]
points2[:, 1] += center[1]
points2[:, 2] += center[2]


angle_rad = np.radians(30)  
angle_rad2 = np.radians(20)
points2 = rotate_xy(points2, center, angle_rad)
points2 = rotate_yz(points2, center, angle_rad2)

# ciculo 3


points3 = np.copy(points2)
center2 = np.array([B_point[0], B_point[1], B_point[2]])
points3[:, 0] += comp[0]
points3[:, 1] += comp[1]
points3[:, 2] += comp[2]
points3 = rotate_xy(points3, comp1, -angle_rad)
points3 = rotate_xz(points3, comp1, -angle_rad2)


ax.scatter(points3[:, 0], points3[:, 1], points3[:, 2], color='black')

ax.scatter(points2[:, 0], points2[:, 1], points2[:, 2], color='black')


# Configuración del gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

ax.set_xlim(-AXIS_LIM, AXIS_LIM)
ax.set_ylim(-AXIS_LIM, AXIS_LIM)
ax.set_zlim(-AXIS_LIM, AXIS_LIM)

plt.show()
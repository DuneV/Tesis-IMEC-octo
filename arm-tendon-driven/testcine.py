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
# parametros = [
#     (0, np.radians(20), np.radians(30), 1, 1, 0.5, 1),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
#     (0, np.radians(20), np.radians(30), 1, 1, 0.5, 2),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
#     (0, np.radians(20), np.radians(30), 1, 1, 0.5, 3),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
#     (0, np.radians(20), np.radians(30), 1, 1, 0.5, 4),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
#     (0, np.radians(20), np.radians(30), 1, 1, 0.5, 5),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
#     (0, np.radians(20), np.radians(30), 1, 1, 0.5, 6),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
#     (0, np.radians(20), np.radians(30), 1, 1, 0.5, 7)  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
#     ]

parametros = [
    (0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0),
    (0, np.radians(-20), np.radians(-30), 1, 1, 0.5, 1)
]

def creation_octopus(parameters):
    N_inicial = ReferenceFrame('N')
    O_inicial = Point('O')
    counter_init = 0
    seccionn = [None] * len(parameters)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    O = [0, 0, 0]
    ax.scatter(*O, color='k', s=50, label='O')
    comps = [None * len(parameters)*3, None*3] 
    for i in range(len(parameters)):
        if counter_init == 0:
            seccionn[i] = Section(N_inicial, O_inicial, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5],parameters[0][6])
            counter_init += 1 
        else:
            seccionn[i] = Section(seccionn[i-1].B, seccionn[i-1].B_point, parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3], parameters[i][4], parameters[i][5], parameters[i][6])
        seccionn[i].update_values(seccionn[i].values)
        



# creation_octopus(parametros)
            


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

comp2 = vector_to_components(seccion.vector1, seccion.N, seccion.values)
comp3 = vector_to_components(seccion.vector2, seccion.N, seccion.values)
comp4 = vector_to_components(seccion.vector3, seccion.N, seccion.values)

comp5 = vector_to_components(seccion2.vector1, seccion.B, seccion2.values)
comp6 = vector_to_components(seccion2.vector2, seccion.B, seccion2.values)
comp7 = vector_to_components(seccion2.vector3, seccion.B, seccion2.values)

centroid = vector_to_components(seccion2.coords_B, seccion.B, seccion2.values)

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


vectors2 = {
    'vector_N2': (seccion2.vector_N, 'magenta'),
    'vector_OA2': (seccion2.vector_OA,'purple'), 
    'vector_AB2': (seccion2.vector_AB, 'g'),
    'vector33': (seccion2.vector3, 'c'),
    'vector22': (seccion2.vector2, 'y'),
    'vector11': (seccion2.vector1, 'b')
}

for label, (vector, color) in vectors2.items():
    if label == 'vector_N2':
        vector_components = vector_to_components(vector, seccion2.N , seccion2.values)
        plot_vector(ax, C_NC, vector_components, color, label)
        C_NC2 = [a + b for a, b in zip(vector_components, C_NC)]
        ax.scatter(*C_NC2, color='g', s=50, label='B2')
    elif label == 'vector_OA2':
        vector_components = vector_to_components(vector, seccion2.N , seccion2.values)
        plot_vector(ax, C_NC, vector_components, color, label)
        A2 = [a + b for a, b in zip(vector_components, C_NC)]
    elif label == 'vector_AB2':
        vector_components = vector_to_components(vector, seccion2.N , seccion2.values)
        plot_vector(ax, A2, vector_components, color, label)
    else:
        vector_components = vector_to_components(vector, seccion2.N , seccion2.values)
        plot_vector(ax, C_NC, vector_components, color, label)



comp2 = [comp2[0] - C_NC[0], comp2[1] - C_NC[1], comp2[2] - C_NC[2]]
comp3 = [comp3[0] - C_NC[0], comp3[1] - C_NC[1], comp3[2] - C_NC[2]]
comp4 = [comp4[0] - C_NC[0], comp4[1] - C_NC[1], comp4[2] - C_NC[2]]

comp5 = [comp5[0] + C_NC[0] - C_NC2[0], comp5[1] + C_NC[1]- C_NC2[1] , comp5[2]+ C_NC[2] - C_NC2[2] ]
comp6 = [comp6[0] + C_NC[0] - C_NC2[0], comp6[1]+ C_NC[1]- C_NC2[1] , comp6[2]+ C_NC[2] - C_NC2[2] ]
comp7 = [comp7[0] + C_NC[0] - C_NC2[0] , comp7[1]+ C_NC[1]- C_NC2[1], comp7[2]+ C_NC[2] - C_NC2[2]]

plot_vector(ax, C_NC, comp2, 'magenta', 'vector_r12')
plot_vector(ax, C_NC, comp3, 'g', 'vector_r22')
plot_vector(ax, C_NC, comp4, 'r', 'vector_r32')

plot_vector(ax, C_NC2, comp5, 'magenta', 'vector_r122')
plot_vector(ax, C_NC2, comp6, 'g', 'vector_r222')
plot_vector(ax, C_NC2, comp7, 'r', 'vector_r322')

# circulo 1

x, y, z = circle_points([0, 0, 0], 0.5, 0)
points = np.vstack((x, y, z)).T
angle_rad_init = np.radians(90)
rotated_points = rotate_x(points, angle_rad_init)

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
# print(points3)
center2 = np.array([C_NC2[0], C_NC2[1], C_NC2[2]])
print(center2)
print(centroid)
points3[:, 0] += centroid[0]
points3[:, 1] += centroid[1]
points3[:, 2] += centroid[2]

print(-2*angle_rad)
print(-2*angle_rad2)

points3 = rotate_xy(points3, center2, -2*angle_rad)
points3 = rotate_yz(points3, center2, -2*angle_rad2)


ax.plot(points3[:, 0], points3[:, 1], points3[:, 2], color='black')
ax.plot(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], color='black')
ax.plot(points2[:, 0], points2[:, 1], points2[:, 2], color='black')


# Configuración del gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
# ax.legend()

ax.set_xlim(-AXIS_LIM, AXIS_LIM)
ax.set_ylim(-AXIS_LIM, AXIS_LIM)
ax.set_zlim(-AXIS_LIM, AXIS_LIM)

plt.show()
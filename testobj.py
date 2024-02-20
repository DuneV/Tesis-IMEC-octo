import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_obj(filename):
    vertices = []
    faces = []
    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                faces.append([int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]])
    return np.array(vertices), np.array(faces)

def reduce_mesh(vertices, faces, factor):
    new_vertices = vertices[::factor]
    # Remapear las caras para reflejar el cambio en los índices de vértices
    new_faces = []
    for face in faces:
        new_face = [index // factor for index in face]
        new_faces.append(new_face)
    return np.array(new_vertices), np.array(new_faces)

# Cargar archivo .obj
vertices, faces = load_obj('section.obj')

# Reducir la malla
reduced_vertices, reduced_faces = reduce_mesh(vertices, faces, factor=2)

# Crear figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Trazar superficie
ax.plot_trisurf(reduced_vertices[:, 0], reduced_vertices[:, 1], reduced_vertices[:, 2], triangles=reduced_faces, cmap='Greys')



# Mostrar gráfico
plt.show()

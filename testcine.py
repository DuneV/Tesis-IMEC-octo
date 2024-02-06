# se importan las librerias
import sympy
from sympy import symbols, pi, cos, sin
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols, Particle, Point
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
#plt.style.use('seaborn')

AXIS_LIM = 2

# se define una sección como una parte del brazo
class Seccion:
    # metodo constructor para varibles de tipo globales

    def __init__(self, N_prev, O_prev, q1_val, q2_val, q3_val, d_A_val, d_B_val, delta_x_val, name_suffix):
        self.q1, self.q2, self.q3 = dynamicsymbols(f'q1_{name_suffix} q2_{name_suffix} q3_{name_suffix}')
        self.d_A, self.d_B, self.delta_x = symbols(f'd_A_{name_suffix} d_B_{name_suffix} delta_x_{name_suffix}')

        self.valores = {
            self.q1: q1_val,
            self.q2: q2_val,
            self.q3: q3_val,
            self.d_A: d_A_val,
            self.d_B: d_B_val,
            self.delta_x: delta_x_val
        }
        
        self.angle = 2 * pi / 3 # dispersión de resorte 2
        self.angle2 = self.angle * 2 # dispersión de resorte 3
        
        # marco de referencia inercial
        self.N = N_prev
        # punto de referencia
        self.O = O_prev
        
        # se define un marco de referencia que rota respecto a N.y
        self.A = self.N.orientnew(f'A_{name_suffix}', 'Axis', (self.q1, self.N.y))
        
        # se define un marco de referecia con respecto a A que rota en los tres ángulos
        self.B = self.A.orientnew(f'B_{name_suffix}', 'Body', (self.q2, self.q1, self.q3), 'XYZ')
        
        # se representa un vector U a lo largo del eje Bx
        self.U = self.B.x
        # se representa un vector B a lo largo del eje Bz 
        self.W = self.B.z

        # aqui sabemos que self.d_A es la distancia de N a A donde tenemos un vector AN_
        self.vector_OA = self.d_A * self.N.y
        # de igual forma con un vector de BA_ en dirección de Ay aqui deberia ser en B NO???? --- Probar
        self.vector_AB = self.d_B * self.B.y
        self.vector_AB = self.vector_AB.express(self.N)

        self.vector_N = self.vector_OA + self.vector_AB
        # se ubica un punto A en el sistema coordenado A
        self.A_point = self.O.locatenew(f'A_point_{name_suffix}', self.vector_OA)
        
        # se ubica un punto B en el sistema coordenado B
        self.B_point = self.A_point.locatenew(f'B_point_{name_suffix}', self.vector_AB)
        
        # se calcula el vector desde el origen hasta el punto B con respecto a N
        self.coords_B = self.B_point.pos_from(self.O).express(self.N)
        
        # se calcula el vector desde el sistema N hasta el punto de flexión de k
        self.vector1 = self.coords_B + (self.delta_x * self.B.x).express(self.N) # es necesario usar el self.N ??

        # self.vector1 = self.vector1.express(self.N)
        
        # encuentra los vectores de cuando esta comprimido con respecto a las ubicaciones de los springs en B
        self.vector2 = self.coords_B + (self.delta_x * (self.B.x * cos(self.angle) + self.B.z * sin(self.angle))).express(self.N)
        self.vector3 = self.coords_B + (self.delta_x * (self.B.x * cos(self.angle2) + self.B.z * sin(self.angle2))).express(self.N)
        
        # encuentro los vectores desd el punto O al punto del springs 1, 2 y 3 con respecto a N (R)
        self.vector_r1 = self.delta_x * self.N.x 
        self.vector_r2 = self.delta_x * (self.N.x * cos(self.angle) + self.N.z * sin(self.angle))
        self.vector_r3 = self.delta_x * (self.N.x * cos(self.angle2) + self.N.z * sin(self.angle2))

    # función de actualización de valores
    def actualizar_valores(self, valores):
        self.vector_OA = self.vector_OA.subs(valores)
        self.vector_AB = self.vector_AB.subs(valores)
        self.coords_B = self.coords_B.subs(valores)
        self.vector1 = self.vector1.subs(valores)
        return self.vector1
    # función de extensiónes 
    def encontrar_exten(self):
        self.vectork1 = self.vector1 - self.vector_r1
        self.vectork2 = self.vector2 - self.vector_r2
        self.vectork3 = self.vector3 - self.vector_r3
        pass

N_inicial = ReferenceFrame('N')
O_inicial = Point('O')

parametros = [
    (0, np.radians(20), np.radians(30), 1, 1, 0.5),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x distancia de O a los spring en mm) d_A + d_B = L
]

seccion = Seccion(N_inicial, O_inicial, 0,  np.radians(20), np.radians(30), 1, 1, 0.5, 0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Función para graficar vectores
def plot_vector(ax, origin, vector, color, label):
    ax.quiver(*origin, *vector, color=color, arrow_length_ratio=0.05, label=label)

# Convertir los vectores Sympy a componentes numéricas
def vector_to_components(vector, subs_dict=None):
    if subs_dict:
        vector = vector.subs(subs_dict)
    return [float(vector.dot(N_inicial.x).evalf()), 
            float(vector.dot(N_inicial.y).evalf()), 
            float(vector.dot(N_inicial.z).evalf())]

# Actualizar los valores de los vectores en la sección
seccion.actualizar_valores(seccion.valores)

# Graficar el punto O (Origen)
O = [0, 0, 0]
ax.scatter(*O, color='k', s=50, label='O')
ax.scatter(*vector_to_components(seccion.coords_B, seccion.valores), color='k', s=50, label='B')
#ax.scatter(*seccion.B_point, color='k', s=50, label='B')
# Graficar vector_OA
vector_OA_components = vector_to_components(seccion.vector_OA, seccion.valores)
plot_vector(ax, O, vector_OA_components, 'r', 'vector_OA')

# Punto A es el destino del vector_OA
A = vector_OA_components

# Graficar vector_AB desde el punto A
vector_AB_components = vector_to_components(seccion.vector_AB, seccion.valores)
plot_vector(ax, A, vector_AB_components, 'g', 'vector_AB')

# Graficar otros vectores desde el origen para simplificar
vectors = {
    'vector1': (seccion.vector1, 'b'),
    'vector2': (seccion.vector2, 'y'),
    'vector3': (seccion.vector3, 'c'),
    'vector_r1': (seccion.vector_r1, 'm'),
    'vector_r2': (seccion.vector_r2, 'orange'),
    'vector_r3': (seccion.vector_r3, 'purple'),
    'vector_N': (seccion.vector_N, 'magenta')
}

for label, (vector, color) in vectors.items():
    if label == 'vector_N':
        vector_components = vector_to_components(vector, seccion.valores)
        plot_vector(ax, O, vector_components, color, label)
        C_NC = vector_components
    else:
        vector_components = vector_to_components(vector, seccion.valores)
        plot_vector(ax, O, vector_components, color, label)

# Función para crear puntos alrededor de un círculo en 3D

def circle_points(center, radius, z, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, z)
    return x, y, z

def circle_points_A(center, radius, z, num_points=100):
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = np.full_like(x, z)
    return np.vstack((x, y, z)).T

def rotate_points(points, angle_rad):
    # Matriz de rotación en el plano XY
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    return np.dot(points, rotation_matrix.T)

# Función para rotar puntos alrededor del eje x y y con las matrices de rotación correspondiente

def rotate_x(points, angle_rad):
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    return np.dot(points, rotation_matrix.T)

def rotate_y(points, angle_rad):
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    return np.dot(points, rotation_matrix.T)

def rotate_xy(points, center, angle_rad):
    # Restar el centro para trasladar los puntos al origen
    translated_points = points - center
    # Matriz de rotación en el plano XY
    rotation_matrix = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0],
        [np.sin(angle_rad), np.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    # Rotar los puntos y trasladarlos de vuelta
    rotated_points = np.dot(translated_points, rotation_matrix.T) + center
    return rotated_points

# circulo 1

x, y, z = circle_points([0, 0, 0], 0.5, 0)
points = np.vstack((x, y, z)).T
angle_rad = np.radians(90)
rotated_points = rotate_x(points, angle_rad)

ax.plot(rotated_points[:,0], rotated_points[:,1], rotated_points[:,2], color='black')

# circulo 2

center = np.array([C_NC[0], C_NC[1], C_NC[2]])
angle_rad = np.radians(45)  # Cambia este ángulo para ver diferentes rotaciones

# Generar puntos para el segundo círculo y aplicar la rotación
points2 = circle_points_A(center[:2], 0.5, center[2])

rotated_points2 = rotate_points(points2, angle_rad)

# Visualizar el círculo antes y después de la rotación
# ax.scatter(points2[:,0], points2[:,1], points2[:,2], color='blue', label='Original')
ax.scatter(rotated_points2[:,0], rotated_points2[:,1], rotated_points2[:,2], color='red', label='Rotated')

# Configuración del gráfico
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

ax.set_xlim(-AXIS_LIM, AXIS_LIM)
ax.set_ylim(-AXIS_LIM, AXIS_LIM)
ax.set_zlim(-AXIS_LIM, AXIS_LIM)

plt.show()
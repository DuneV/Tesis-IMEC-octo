# se importan las librerias
import sympy
from sympy import symbols, pi, cos, sin
from sympy.physics.mechanics import ReferenceFrame, dynamicsymbols, Particle, Point
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
        self.vector_AB = self.d_B * self.A.y

        # se ubica un punto A en el sistema coordenado A
        self.A_point = self.O.locatenew(f'A_point_{name_suffix}', self.vector_OA)
        
        # se ubica un punto B en el sistema coordenado B
        self.B_point = self.A_point.locatenew(f'B_point_{name_suffix}', self.vector_AB)
        
        # se calcula el vector desde el origen hasta el punto B con respecto a N
        self.coords_B = self.B_point.pos_from(self.O).express(self.N)
        
        # se calcula el vector desde 
        self.vector1 = self.coords_B + self.delta_x * self.B.x
        self.vector1 = self.vector1.express(self.N)
        
        self.vector2 = self.coords_B + self.delta_x * (self.B.x * cos(self.angle) + self.B.z * sin(self.angle))
        self.vector3 = self.coords_B + self.delta_x * (self.B.x * cos(self.angle2) + self.B.z * sin(self.angle2))

        self.vector_r1 = self.delta_x * self.N.x
        self.vector_r2 = self.delta_x * (self.N.x * cos(self.angle) + self.N.z * sin(self.angle))
        self.vector_r3 = self.delta_x * (self.N.x * cos(self.angle2) + self.N.z * sin(self.angle2))


    def actualizar_valores(self, valores):
        self.vector_OA = self.vector_OA.subs(valores)
        self.vector_AB = self.vector_AB.subs(valores)
        self.coords_B = self.coords_B.subs(valores)
        self.vector1 = self.vector1.subs(valores)
        return self.vector1
    
    def encontrar_exten(self):
        self.vectork1 = self.vector1 - self.vector_r1
        self.vectork2 = self.vector2 - self.vector_r2
        self.vectork3 = self.vector3 - self.vector_r3
        pass


N_inicial = ReferenceFrame('N')
O_inicial = Point('O')

parametros = [
    (10, 20, 30, 1, 1, 0.5),  # (q1 (y), q2(z), q3(x), d_A, d_B, delta_x) d_A + d_B = L
    (15, 25, 35, 1, 1, 0.5),
]

secciones = []

for i, (q1_val, q2_val, q3_val, d_A_val, d_B_val, delta_x_val) in enumerate(parametros):
    seccion = Seccion(N_inicial if i == 0 else secciones[-1].B, O_inicial if i == 0 else secciones[-1].B_point, q1_val, q2_val, q3_val, d_A_val, d_B_val, delta_x_val, name_suffix=str(i))
    vector1_actualizado = seccion.actualizar_valores(seccion.valores)
    print(f'Sección {i}: Vector1 = {vector1_actualizado}')
    secciones.append(seccion)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def extraer_componentes(vector, valores, N_ref):
    # Sustituye los símbolos en el vector por sus valores correspondientes
    vector_sustituido = vector.subs(valores)
    
    # Evalúa cada componente a un valor flotante
    x = vector_sustituido.dot(N_ref.x).evalf()
    y = vector_sustituido.dot(N_ref.y).evalf()
    z = vector_sustituido.dot(N_ref.z).evalf()
    
    return float(x), float(y), float(z)

# Inicialización de la figura de Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Iterar sobre cada sección para graficar los vectores
for i, seccion in enumerate(secciones):
    # Extraer componentes y graficar el vector_OA desde el origen
    x0, y0, z0 = extraer_componentes(seccion.vector_OA, seccion.valores, seccion.N)
    ax.quiver(0, 0, 0, x0, y0, z0, color='b', length=1, normalize=False)
    
    # Sumar vector_OA y vector1 para obtener la posición final de vector1 en el sistema N
    vector_total = seccion.vector_OA + seccion.vector1
    x_total, y_total, z_total = extraer_componentes(vector_total, seccion.valores, seccion.N)
    
    # Graficar el vector total desde el origen para mostrar la secuencia contigua
    ax.quiver(0, 0, 0, x_total, y_total, z_total, color='r', length=1, normalize=False)

# Establecer límites y etiquetas de los ejes
ax.set_xlim([0, 10])
ax.set_ylim([0, 10])
ax.set_zlim([0, 10])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Mostrar el gráfico
plt.show()
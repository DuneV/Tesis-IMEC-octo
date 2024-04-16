from sympy.physics.mechanics import dynamicsymbols
from sympy import symbols, pi, cos, sin

class Section:
    
    '''
        The constructor method take the 
        N -> Reference frame
        O -> Initial Point
        q_1 -> y axis angle rotation
        q_2 -> z axis angle rotation
        q_3 -> x axis angle rotation
        d_A -> origin of module distance to the universal joint
        d_B -> universal joint to other cardan
        delta_x -> distance O to the springs
        name_suffix -> Id module
    '''
    
    def __init__(self, N_prev, O_prev, q1_val, q2_val, q3_val, d_A_val, d_B_val, delta_x_val, name_suffix):

        self.q1, self.q2, self.q3 = dynamicsymbols(f'q1_{name_suffix} q2_{name_suffix} q3_{name_suffix}')

        self.d_A, self.d_B, self.delta_x = symbols(f'd_A_{name_suffix} d_B_{name_suffix} delta_x_{name_suffix}')

        self.values = {
            self.q1: q1_val,
            self.q2: q2_val,
            self.q3: q3_val,
            self.d_A: d_A_val,
            self.d_B: d_B_val,
            self.delta_x: delta_x_val
        }
        
        self.angle = 2 * pi / 3 
        self.angle2 = self.angle * 2 # change in case to add more springs to the module
        
        # reference frame
        self.N = N_prev
        # reference point
        self.O = O_prev
        
        # define a frame that rotates with the q_1 angle considering N.y
        self.A = self.N.orientnew(f'A_{name_suffix}', 'Axis', (self.q1, self.N.y))
        
        # define a frame that rotates with q_1, q_2 and q_3.
        self.B = self.A.orientnew(f'B_{name_suffix}', 'Body', (self.q2, self.q1, self.q3), 'XYZ')
        
        # vector with direction B.x
        self.U = self.B.x
        # vector with direction B.z
        self.W = self.B.z

        # vector between N reference and the joint point A
        self.vector_OA = self.d_A * self.N.y

        # vector with the orientation from B to the next section
        self.vector_AB = self.d_B * self.B.y

        # self.vector_AB = self.vector_AB.express(self.N) this does not apply changes on the vector

        # obtain the vector from the two sections
        self.vector_N = self.vector_OA + self.vector_AB

        # it gives position of the cardan joint
        self.A_point = self.O.locatenew(f'A_point_{name_suffix}', self.vector_OA)
        
        # it gives position of the other section
        self.B_point = self.O.locatenew(f'B_point_{name_suffix}', self.vector_N)
        
        # calculate the coordenates of the next joint

        # self.coords_B = self.B_point.pos_from(self.O).express(self.N) # another option ---> case to have more complicated system
        self.coords_B = self.vector_N.express(self.N)

        # origin to spring
        self.vector1 = self.coords_B + (self.delta_x * self.B.x).express(self.N) # neccesary to express it on N

        # self.vector1 = self.vector1.express(self.N)
        
        # it found the vectors from the B part to the springs
        self.vector2 = self.coords_B + (self.delta_x * (self.B.x * cos(self.angle) + self.B.z * sin(self.angle))).express(self.N)
        self.vector3 = self.coords_B + (self.delta_x * (self.B.x * cos(-self.angle) + self.B.z * sin(-self.angle))).express(self.N)
        
        # it found the vectors from the O part to the springs respect to N (R)
        self.vector_r1 = self.delta_x * self.N.x 
        self.vector_r2 = self.delta_x * (self.N.x * cos(self.angle) + self.N.z * sin(self.angle))
        self.vector_r3 = self.delta_x * (self.N.x * cos(-self.angle) + self.N.z * sin(-self.angle))

    # update the values
    def update_values(self, values):
        self.vector_OAv = self.vector_OA.subs(values)
        self.vector_AB = self.vector_AB.subs(values)
        self.coords_B = self.coords_B.subs(values)
        self.vector1 = self.vector1.subs(values)
        return [self.vector_OAv, self.vector_OA]
    
    # found the extensión of the springs 

    def spring_vectors(self):
        self.vectork1 = self.vector1 - self.vector_r1
        self.vectork2 = self.vector2 - self.vector_r2
        self.vectork3 = self.vector3 - self.vector_r3
        pass
    
    def documentation(self):
        print(Section.__doc__)



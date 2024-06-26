import numpy as np


def plot_vector(ax, origin, vector, color, label): # vector is the value on each axis
    origin_x, origin_y, origin_z = origin
    vector_x, vector_y, vector_z = vector
    ax.quiver(origin_x, origin_y, origin_z, vector_x, vector_y, vector_z, color=color, arrow_length_ratio=0.05, label=label)

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

def rotate_xz(points, center, angle_rad):
    # Restar el centro para trasladar los puntos al origen
    translated_points = points - center
    # Matriz de rotación en el plano XZ
    rotation_matrix = np.array([
        [np.cos(angle_rad), 0, np.sin(angle_rad)],
        [0, 1, 0],
        [-np.sin(angle_rad), 0, np.cos(angle_rad)]
    ])
    # Rotar los puntos y trasladarlos de vuelta
    rotated_points = np.dot(translated_points, rotation_matrix.T) + center
    return rotated_points

def rotate_yz(points, center, angle_rad):
    # Restar el centro para trasladar los puntos al origen
    translated_points = points - center
    # Matriz de rotación en el plano YZ
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle_rad), -np.sin(angle_rad)],
        [0, np.sin(angle_rad), np.cos(angle_rad)]
    ])
    # Rotar los puntos y trasladarlos de vuelta
    rotated_points = np.dot(translated_points, rotation_matrix.T) + center
    return rotated_points

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

def vector_to_components(vector, frame, subs_dict=None):
    if subs_dict:
        vector = vector.subs(subs_dict)
    return [float(vector.dot(frame.x).evalf(subs=subs_dict)), 
            float(vector.dot(frame.y).evalf(subs=subs_dict)), 
            float(vector.dot(frame.z).evalf(subs=subs_dict))]

def vector_magnitude(vector):
    return np.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
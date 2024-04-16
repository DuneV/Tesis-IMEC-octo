import numpy as np

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
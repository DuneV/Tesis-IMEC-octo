import requests
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from stl import mesh

# Importing rotation matrix
from lib.rotation import RPY2XYZ, D2R

# ESP32 IP
ip = '192.168.4.1'
url_x = f'http://{ip}/AngleX'
url_y = f'http://{ip}/AngleY'
url_z = f'http://{ip}/AngleZ'

# Mesh file
path_mesh = 'resources/RockHead.stl'
mf = mesh.Mesh.from_file(path_mesh)
com = mf.vectors.mean(axis=(0, 1))

def rotate_mesh(mesh, angles):
    phi, theta, psi = angles
    rotation_matrix = RPY2XYZ([D2R(phi), D2R(theta), D2R(psi)])
    for i in range(len(mesh.vectors)):
        for j in range(3):
            mesh.vectors[i][j] = np.dot(rotation_matrix, mesh.vectors[i][j] - com) + com

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial plot
mesh_collection = ax.add_collection3d(Poly3DCollection(mf.vectors))

def get_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return None

def get_gyros():
    angle_x = get_data(url_x)
    angle_y = get_data(url_y)
    angle_z = get_data(url_z)
    
    return [angle_x, angle_y, angle_z]


def update_point(n):
    try:
        phi, theta, psi = map(float, get_gyros())
        angles = (phi, theta, psi)
        rotate_mesh(mf, angles)
        error_display.set_text("")
    except Exception as e:
        error_display.set_text(f"Error: {e}")

    ax.clear()
    ax.add_collection3d(Poly3DCollection(mf.vectors))
    smallDetails(ax)
    return mesh_collection,


def smallDetails(ax):
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-20, 20])
    ax.set_ylim([-20, 20])
    ax.set_zlim([-20, 20])
    ax.view_init(elev=20., azim=30)

state_display = ax.text2D(0.07, 1.0, "", color='green', transform=ax.transAxes)
error_display = ax.text2D(0.07, 0.95, "", color='red', transform=ax.transAxes)

ani = animation.FuncAnimation(fig, update_point, interval=500) # Adjust interval as needed

# Show the plot
plt.show()

import requests
import time

# Reemplaza '<IP>' con la dirección IP de tu ESP32
ip = '192.168.4.1'
url_x = f'http://{ip}/AngleX'
url_y = f'http://{ip}/AngleY'
url_z = f'http://{ip}/AngleZ'

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

while True:
    angle_x = get_data(url_x)
    angle_y = get_data(url_y)
    angle_z = get_data(url_z)

    if angle_x is not None and angle_y is not None and angle_z is not None:
        print(f"Angle X: {angle_x} *")
        print(f"Angle Y: {angle_y} *")
        print(f"Angle Z: {angle_z} *")

    time.sleep(0.05)  # Intervalo de 0.5 segundos entre solicitudes

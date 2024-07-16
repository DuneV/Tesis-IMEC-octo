import requests
import time

# Dirección IP del ESP32
ip = '192.168.4.1'
url = f'http://{ip}/angles'

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
    data = get_data(url)
    if data:
        try:
            angle_x, angle_y, angle_z = map(float, data.split(';'))
        except ValueError as e:
            print(f"Data parsing error: {e}")

    if angle_x is not None and angle_y is not None and angle_z is not None:
        print(f"Angle X: {angle_x} *")
        print(f"Angle Y: {angle_y} *")
        print(f"Angle Z: {angle_z} *")

    time.sleep(0.05)  # Intervalo de 0.5 segundos entre solicitudes

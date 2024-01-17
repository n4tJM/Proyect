import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
import matplotlib.pyplot as plt
import cv2
from PIL import Image

env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
env.reset()

# Definir la forma de la matriz para almacenar las imágenes
num_steps = 10
image_shape = env.observation_space.shape
image_matrix = np.zeros((num_steps, *image_shape), dtype=np.uint8)

for step in range(num_steps):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Guardar la observación en la matriz
    image_matrix[step] = obs

    if done:
        env.reset()

env.close()

print(image_matrix)


def convertir_a_escala_de_grises(matriz_de_imagenes):
    """
    Convierte cada imagen en escala de grises.

    Parametros:
    - matriz_de_imagenes: numpy.ndarray
        Matriz de imágenes a convertir.

    Returns:
    - matriz_escala_de_grises: numpy.ndarray
        Matriz de imágenes en escala de grises.
    """
    num_frames, height, width, _ = matriz_de_imagenes.shape
    matriz_escala_de_grises = np.zeros((num_frames, height, width), dtype=np.uint8)

    for i in range(num_frames):
        # Convierte la imagen a escala de grises
        imagen_escala_de_grises = cv2.cvtColor(matriz_de_imagenes[i], cv2.COLOR_RGB2GRAY)
        matriz_escala_de_grises[i] = imagen_escala_de_grises

    return matriz_escala_de_grises

matriz_en_escala_de_grises = convertir_a_escala_de_grises(image_matrix)

#print(matriz_en_escala_de_grises)

plt.imshow(matriz_en_escala_de_grises[0], cmap='gray')
plt.title("Imagen en Escala de Grises Ponderado")
plt.show()

def disminuir_imagen(img_matriz):
    """
    Cambia el tamaño de la imagen de 255 por 255 a 50 po 50
    """
    image = Image.fromarray(img_matriz[0]) # Convierte el array NumPy a formato de imagen PIL
    new_size = (50, 50)

    resized_image = image.resize(new_size)
    resized_array = np.array(resized_image) 
    return resized_array

matriz_final = disminuir_imagen(matriz_en_escala_de_grises)
plt.imshow(matriz_final, cmap='gray')
plt.title("Imagen en Escala de Grises Ponderado")
plt.show()
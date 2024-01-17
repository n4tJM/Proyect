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
env.reset()      #30 fotogramas por segundo
for step in range(2500):
    action = env.action_space.sample() # remplazar por la red neuronal 
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    # print(obs) # sumar todos y dividirlos entre 255 (para convertirlo en escala de grises)
    #print(f"Step: {step}, Action: {action}, Reward: {reward}, Done: {done}, Info: {info}")
    # la red recibe: posicion de mario e imagen

    if done:
       env.reset()

env.close()

# -----------------------------NEW DOCUMENT----------------------------------

def generar_imagenes_rewards(num_steps:int):
    """
    Genera los frames y los rewards
    Parametros:
    - num_steps: int
        Número de pasos

    Returns:
    - Matriz de imagenes (frames)
    - Vector con los rewards
    """
    env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True)
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = True
    env.reset()

    # Definir la forma de la matriz para almacenar las imágenes
    image_shape = env.observation_space.shape
    image_matrix = np.zeros((num_steps, *image_shape), dtype=np.uint8)
    rewards = np.zeros((num_steps), dtype=np.uint8)

    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Guardar la observación en la matriz
        image_matrix[step] = obs
        rewards[step] = reward


        if done:
            env.reset()

    env.close()
    return image_matrix, rewards


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


def disminuir_imagen(img_matriz):
    """
    Cambia el tamaño de cada imagen en la matriz de 255 por 255 a 50 por 50
    """
    new_size = (50, 50)
    resized_images = []

    for img in img_matriz:
        image = Image.fromarray(img)  # Convierte el array NumPy a formato de imagen PIL
        resized_image = image.resize(new_size)
        resized_array = np.array(resized_image)
        resized_images.append(resized_array)

    return np.array(resized_images)

imagenes, puntaje = generar_imagenes_rewards(50)
imagenes_grises = convertir_a_escala_de_grises(imagenes)
imagenes_5050 = disminuir_imagen(imagenes_grises)

plt.imshow(imagenes_5050[-1], cmap='gray')
plt.title("Imagen en Escala de Grises 50 x 50")
plt.show()
#Multi Layer Perceptivo - RN - Computação Natural
# 30/01/23


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()

# print(train_images.shape)   
# print(test_images.shape)
# print(train_images[0])
# print(train_labels)

categorias = ['camiseta', 'calça', 'sueter', 'vestido', 'casaco', 'sandalia', 'camisa', 'tenis', 'bolsa', 'bota']

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Transformando em escala 0-1
#setar a imagem pra ficar melhor pro ReLU e sigmoid (escala de cinza)
train_images = train_images/255.0
test_images = test_images/255.0

# Plotando menor as 25 primeiras imagens
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap = plt.cm.binary)
  plt.xlabel(categorias[train_labels[i]])
plt.show()

#Observação: modo como formata os dados para as funções de ativação 
# Sigmóide (0 - 1) e tangente hiperbólica (-1 - 1)

#Orientado à objetos 
modelo = keras.Sequential()
modelo.add(keras.layers.Flatten(input_shape = (28,28))) #input
modelo.add(keras.layers.Dense(128, activation = 'relu')) #camada intermediária
modelo.add(keras.layers.Dense(64, activation = 'relu'))
modelo.add(keras.layers.Dense(10, activation = 'softmax')) #saída 
#n = número de neurônios

#Compilar o modelo 
modelo.compile(optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy']
)

#treinar o modelo
modelo.fit(train_images, train_labels, epochs = 10)


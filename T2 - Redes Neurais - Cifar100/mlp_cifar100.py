import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt

#fine: apenas a classe da imagem
#coarse: a superclasse da imagem
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

#normalização 
train_images = train_images/255.0 
test_images = test_images/255.0 

#armazenamento das classes
lista = []
arquivo = open("fine.txt", "r")
dado = arquivo.readlines()
	
for linha in dado: 
	classe = linha.split(",")
	lista.append(classe)
	
classes = lista[0]

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


#Orientado à objetos 
modelo = keras.Sequential()
modelo.add(keras.layers.Flatten(input_shape = (32,32,3))) #input do Cifar100
modelo.add(keras.layers.Dense(1024, activation = 'relu')) #camadas intermediárias
modelo.add(keras.layers.Dense(512, activation = 'relu')) 
modelo.add(keras.layers.Dense(256, activation = 'relu')) 
modelo.add(keras.layers.Dense(128, activation = 'relu')) 
modelo.add(keras.layers.Dense(100, activation = 'softmax')) #saída 

#Compilar o modelo 
modelo.compile(optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = ['accuracy']
)

#treinar o modelo
modelo.fit(train_images, train_labels, epochs = 50)

# Plotando menor as 25 primeiras imagens
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap = plt.cm.binary)
  plt.xlabel(classes[int(train_labels[i])])
plt.show()

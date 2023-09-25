import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalizar pixels para terem valores entre 0 e 1
train_images, test_images = train_images / 255.0, test_images / 255.0

#armazenamento das classes
lista = []
arquivo = open("fine.txt", "r")
dado = arquivo.readlines()

for linha in dado:
	classe = linha.split(",")
	lista.append(classe)

classes = lista[0]

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(classes[int(train_labels[i])])
plt.show()

#Criação da base convolucional 
modelo = models.Sequential()
modelo.add(layers.Conv2D(256, (3, 3), activation='relu', input_shape=(32, 32, 3)))
modelo.add(layers.MaxPooling2D((2, 2)))

modelo.add(layers.Conv2D(128, (3, 3), activation='relu'))
modelo.add(layers.MaxPooling2D((2, 2)))

modelo.add(layers.Conv2D(100, (3, 3), activation='relu'))
modelo.add(layers.MaxPooling2D((2, 2)))

modelo.add(layers.Conv2D(100, (3, 3), activation='relu'))

#Exibir a arquitetura do modelo
#model.summary()

#Topo -> Camadas densas 
modelo.add(layers.Flatten())
modelo.add(layers.Dense(100, activation='relu'))
modelo.add(layers.Dense(100, activation = 'softmax')) #softmax normaliza os dados

#Comparar diferenças 
#model.summary()

#Compilar e testar o modelo 
modelo.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

historico = modelo.fit(train_images, train_labels, epochs=30,
                    validation_data=(test_images, test_labels))


#Avaliação do modelo
plt.plot(historico.history['accuracy'], label='accuracy')
plt.plot(historico.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = modelo.evaluate(test_images,  test_labels, verbose=2)


print(test_acc)
#0.7192000150680542 acurácia de aprox 70% com Cifar10, 3 camadas de 32 e 64 neurônios

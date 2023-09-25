from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, LeakyReLU, Reshape, Conv2DTranspose
from keras.datasets.mnist import load_data
from numpy import ones

(train_dados, train_rotulos), (teste_dados, teste_rotulos) = load_data()
train_dados = train_dados/255.0
teste_dados = teste_dados/255.0

#Rede detectora
def detectora(input_shape = (28, 28, 1)):
  model = Sequential()
  model.add(Conv2D(64, (3, 3), strides = (2, 2), padding = 'same', input_shape = input_shape))#conv
  model.add(LeakyReLU(alpha = 0.2))
  model.add(Dropout(0.4))
  model.add(Conv2D(64, (3, 3), strides = (2, 2), padding = 'same'))
  model.add(Dropout(0.4))#conv
  model.add(Flatten())#mlp
  model.add(Dense(1, activation = 'sigmoid')) #mlp
  opt = Adam(lr = 0.0002, beta_1 = 0.5)
  model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
  return model

#Rede geradora
def geradora(dim_latente):
  modelo = Sequential()
  num_neuronios = 128 * 7 * 7
  modelo.add(Dense(num_neuronios, input_shape = dim_latente))
  modelo.add(LeakyReLU(alpha = 0.2))
  modelo.add(Reshape((7, 7, 128))) # image 7x7
  modelo.add(Conv2DTranspose(128, (4, 4), strides = (2, 2), padding = 'same')) # imagem 14x14
  modelo.add(LeakyReLU(alpha = 0.2))
  modelo.add(Conv2DTranspose(123, (4, 4), strides = (2, 2), padding = 'same')) # 28x28
  modelo.add(LeakyReLU(alpha = 0.2))
  modelo.add(Conv2D(1, (7, 7), activation = 'sigmoid', padding = 'same'))
  return modelo

#Generative Adversarial Network
def gan(modelo_g, modelo_d):
  modelo_d.trainable = False
  modelo = Sequential()
  modelo.add(modelo_g)
  modelo.add(modelo_d)
  opt = Adam(lr = 0.0002, beta_1 = 0.5)
  modelo.complile(loss = 'binary_crossentropy', optimizer = opt)
  return modelo

#Treino
def treina(modelo_g, modelo_d, modelo_gan, dados, dim_latente, num_epocas = 100, tam_batch = 256):
  batches_por_epoca = int(dados.shape[0] / tam_batch)
  metade_batch = tam_batch/2
  for i in range(num_epocas):
    for j in range(batches_por_epoca):
      dados_reais, rotulos_reais = gera_reais(dados, metade_batch) # seleciona pedacos da base de dados
      modelo_d.train_on_batch(dados_reais, rotulos_reais)
      dados_falsos, rotulos_falsos = gera_dados_falsos(metade_batch) # criar imagens aleatorias com 0 e 1, pegar imagens da base que nao sao o objetivo, inserir dados da geradora (mais tarde)
      modelo_d.train_on_batch(dados_falsos, rotulos_falsos)

      dados_gan = gera_dimensao_latente(dim_latente, tam_batch)
      rotulos_gan = ones((tam_batch, 1))
      gan_perda = modelo_gan.train_on_batch(dados_gan, rotulos_gan)


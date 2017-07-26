# encoding: UTF-8
# Copyright 2017 Udacity.com
# Authored by Daniel Rodrigues Loureiro (drlschilling@gmail.com)
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


############################################################################
#                                                                          #
# Neste segundo exemplo de construcao de um modelo de classificacao iremos #
# introduzir alguns conceitos que deixam nosso modelo mais complexo, porem #
# com uma taxa de erro reduzida quando utilizamos camadas convolucionais   #
# com tratamento de Dropout e MaxPooling.                                  # 
#                                                                          #               
############################################################################

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from vis.utils import utils
import numpy as np
from matplotlib import pyplot as plt

import keras_mnist_vis

# Usaremos o metodo de Dropout, um metodo 
# de regularizacao, para diminuir ao maximo 
# nossa chance de overfitting (https://keras.io/layers/core/#dropout)
from keras.layers import Dropout

# Para reduzirmos o dado de input a 
# uma representacao do total de pixels (https://keras.io/layers/core/#flatten)
from keras.layers import Flatten

##########################################################################
#                                                                        #
# A camada convolucional cria um kernel de convoluçao na entrada da      #
# camada para produzir um tensor na saida.                               #
#                                                                        #
# Uma rede convolucional tem quatro etapas:                              #
#                                                                        #
# 1) Convolucao                                                          #    
#                                                                        #
# A convoluçao funciona como um rotulador do dado de entrada             #
# sempre se referindo ao que o modelo aprendeu anteriormente.            #
#                                                                        #    
# 2) Subamostragem                                                       #     
#                                                                        #
# Os inputs da camada convolucional podem ter sensibilidade              # 
# de seus filtros reduzida frente ao ruido, e esse processo              #    
# chamamos de subamostragem.                                             #
#                                                                        #
# 3) Ativacao                                                            #
#                                                                        #    
# A camada de ativacao controla como o sinal flui de uma camada          #
# para outra, emulando como nossos neuronios sao ativados.               #    
#                                                                        #
# 4) Conexao total (fully connected)                                     #    
#                                                                        #
# As camadas que ficam por ultimo na rede estao                          #
# totalmente conectadas, o que significa que os neurônios                #
# das camadas precedentes estao conectados com os subsequentes.          #    
#                                                                        #
##########################################################################
from keras.layers.convolutional import Conv2D

# A camada de pooling e frequentemente usada em redes neurais tem como
# objetivo reduzir progressivamente o tamanho espacial da representacao 
# e a complexidade computacional da rede.
from keras.layers.convolutional import MaxPooling2D

from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# Construimos nossos subconjuntos de treinamento e teste.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Como estamos trabalhando em escala de cores cinza podemos
# definir a dimensao do pixel como sendo 1.
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# Normalizamos nossos dados de acordo com variacao da
# escala de cinza.
X_train = X_train / 255
X_test = X_test / 255

# Aplicamos a solucao de one-hot-encoding para
# classificacao multiclasses.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Numero de tipos de digitos encontrados no MNIST.
num_classes = y_test.shape[1]

def cnn_model():
    model = Sequential()

    # A Convolution2D sera a nossa camada de entrada. Podemos observar que ela possui 
    # 32 mapas de features com tamanho de 5 × 5 e 'relu' como funcao de ativacao. 
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # A camada MaxPooling2D sera nossa segunda camada onde teremos um amostragem de 
    # dimensoes 2 × 2.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Na nossa camada de regularizacao usamos o metodo de Dropout
    # excluindo 30% dos neuronios na camada, diminuindo nossa chance de overfitting.
    model.add(Dropout(0.3))

    # Usamos a camada de Flatten para converter nossa matriz 2D
    # numa representacao a ser processada pela fully connected.
    model.add(Flatten())

    # Camada fully connected com 128 neuronios e funcao de ativacao 'relu'.
    model.add(Dense(128, activation='relu'))
    
    # Nossa camada de saida possui o numero de neuronios compativel com o 
    # numero de classes a serem classificadas, com uma funcao de ativacao
    # do tipo 'softmax'.
    model.add(Dense(num_classes, activation='softmax', name='preds'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = cnn_model()

# O metodo summary revela quais sao as camadas
# que formam o modelo, seus formatos e o numero
# de parametros envolvidos em cada etapa.
model.summary()

# Processo de treinamento do modelo. 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=2)

# Avaliacao da performance do nosso primeiro modelo.
scores = model.evaluate(X_test, y_test, verbose=0)
keras_mnist_vis.keras_digits_vis(model, X_test, y_test)
print("Erro de: %.2f%%" % (100-scores[1]*100))

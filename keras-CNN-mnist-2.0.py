# encoding: UTF-8
# Copyright 2017 Udacity.com
#
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

##############################################################################
#                                                                            #
# Neste terceiro exemplo buscamos criar um modelo em que o desenho tenha     #
# o objetivo de diminuir ao maximo o erro de classificacao e retornando a    #
# maior taxa de acuracia possivel. Diferentemente do exemplo anterior        #
# teremos duas camadas convolucionais,  duas camadas de pooling e duas       # 
# fully connected.Podemos a partir deste exemplo revelar o quanto a          #
# arquitetura do modelo pode influenciar no resultado final da classificacao #                                   
#                                                                            #                
#############################################################################

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from vis.visualization import visualize_saliency
from vis.utils import utils
import numpy as np
from keras import activations
from matplotlib import pyplot as plt
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import keras_mnist_vis 
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

def deeper_cnn_model():
    model = Sequential()

    # A Convolution2D sera a nossa camada de entrada. Podemos observar que ela possui 
    # 30 mapas de features com tamanho de 5 × 5 e 'relu' como funcao de ativacao. 
    model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))

    # A camada MaxPooling2D sera nossa segunda camada onde teremos um amostragem de 
    # dimensoes 2 × 2.
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Uma nova camada convolucional com 15 mapas de features com dimensoes de 3 × 3 
    # e 'relu' como funcao de ativacao. 
    model.add(Conv2D(15, (3, 3), activation='relu'))

    # Uma nova subamostragem com um pooling de dimensoes 2 x 2.
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Uma camada de Dropout com probabilidade de 20%
    model.add(Dropout(0.2))

    # Uma camada de Flatten preparando os dados para a camada fully connected. 
    model.add(Flatten())

    # Camada fully connected de 128 neuronios.
    model.add(Dense(128, activation='relu'))

    # Seguida de uma nova camada fully connected de 64 neuronios
    model.add(Dense(64, activation='relu'))

    # A camada de saida possui o numero de neuronios compativel com o 
    # numero de classes a serem classificadas, com uma funcao de ativacao
    # do tipo 'softmax'.
    model.add(Dense(num_classes, activation='softmax', name='preds'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = deeper_cnn_model()

# O metodo summary revela quais sao as camadas
# que formam o modelo, seus formatos e o numero
# de parametros envolvidos em cada etapa.
model.summary()

# Processo de treinamento do modelo. 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200)

# Avaliacao da performance do nosso primeiro modelo.
scores = model.evaluate(X_test, y_test, verbose=0)
keras_mnist_vis.keras_digits_vis(model, X_test, y_test)
print("Erro de: %.2f%%" % (100-scores[1]*100))

 
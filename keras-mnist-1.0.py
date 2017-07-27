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
# Aqui vamos construir nosso primeiro exemplo de modelo de classificacao   #
# do nosso conjunto de dados MNIST.                                        # 
#                                                                          # 
# Este exemplo é uma basica demonstracao com uma simples rede neural       #
# de apenas uma camada, e que poderemos utilizar como comparativo com      #
# modelos mais complexos construidos a seguir.                             #              
############################################################################

# Importamos o conjunto de dados MNIST
from keras.datasets import mnist

# Para visualizarmos bem a sequencia de camadas do modelo 
# vamos usar o modulo do Keras chamado Sequential 
# (https://keras.io/getting-started/sequential-model-guide/)
from keras.models import Sequential

# Como estamos construindo um modelo simples vamos utilizar
# camadas densas, que sao simplesmente camadas onde cada unidade
# ou neuronio estara conectado a cada neurônio na proxima camada.
from keras.layers import Dense

# Modulo do Keras responsavel por varias rotinas de pre-processamento 
# (https://keras.io/utils/).
from keras.utils import np_utils

# Assim como vimos anteriormente em nosso exemplo de visualizacao
# aqui estamos carregando o conjunto de dados em subconjuntos de 
# treinamento e teste.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#################################################################################
#                                                                               # 
# Nosso conjunto de treinamento é aqui estruturado em forma de uma matriz       # 
# tridimensional que representa as dimensoes de largura, altura e               #
# numero de pixels uma imagem. Como estamos trabalhando com um modelo que       #
# possui uma camada de neuronios sera necessario mudarmos a dimensionalidade    #
# desta representacao para que ela possa servir como dado de entrada em nosso   #
# modelo, para isso reduzirmos essas dimensoes na forma de um vetor de pixels   #
# que possa representar essa imagem.                                            #
#                                                                               #
# No caso em questao temos que ter em mente que uma imagem de tamanho 28 × 28   #
# tera valores de entrada de 784 pixels. Para atingirmos essa representacao     #
# vamos usar a funcao reshape nas matrizes de treinamento e teste.              #
#                                                                               #
################################################################################# 
num_pixels = X_train.shape[1] * X_train.shape[2]

# Com o intuito de amenizar o uso de memoria podemos atribuir um nivel precisao dos 
# valores de pixel com sendo 32 bits (float32)
X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32')


# Podemos normaliza os valores de pixels para o intervalo 0 e 1 
# dividindo cada valor pelo máximo de 255, visto que os valores 
# de pixel estao escala de cinza entre 0 e 255.  
X_train = X_train / 255
X_test = X_test / 255

# Como estamos trabalhando com um problema de classificacao
# multiclasses, pois temos varios tipos de digitos, vamos 
# represanta-los em categorias usando a metodologia de 
# one-hot-encoding aqui representada pela funcao to_categorical.
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Numero de tipos de digitos encontrados no MNIST.
num_classes = y_test.shape[1]

# Modelo basico de uma camada onde inicializamos um modelo sequencial
# com suas funcoes de ativacao, e o compilamos usando gradiente descendente e
# acuracia como metrica.

def base_model():
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax', name='preds'))
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	return model


model = base_model()

# O metodo summary revela quais sao as camadas
# que formam o modelo, seus formatos e o numero
# de parametros envolvidos em cada etapa.
model.summary()

# Processo de treinamento do modelo. 
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=100, verbose=2)

# Avaliacao da performance do nosso primeiro modelo.
scores = model.evaluate(X_test, y_test, verbose=0)
print("Erro de: %.2f%%" % (100-scores[1]*100))

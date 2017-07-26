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
# Keras possui um metodo de facil acesso ao conjunto de dados MNIST.       #
# Ele e baixado de forma automatica e armazenado num diretorio             #
# ~ / .keras / datasets / mnist.pkl.gz em sua maquina.                     #
#                                                                          #
# Precisamos conhecer a verdadeira face desses dados, para isso            #
# vamos testar a seguir como podemos carregar esse conjunto de dados       #
# e visualizar as suas quatro primeiras amostras.                          #
#                                                                          #
############################################################################


# Importamos da biblioteca Keras o conjunto de dados MNIST 
# sendo que a biblioteca oferece diversas outras opcoes de 
# conjuntos de dados, a saber (https://keras.io/datasets/).
from keras.datasets import mnist 

# Importamos a biblioteca de construcao de graficos matplotlib
# para visualizarmos as amostras.
import matplotlib.pyplot as plt

# Atraves do objeto 'mnist' usamos a funcao load_data 
# e construimos nossos subconjuntos de treinamento e teste.
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Cada subplot construindo sera a representacao 
# de uma amostra. Aqui estamos extraindo quatro exemplos
# do nosso conjuntos de treinamento. 

num_samples = 4

for i in range(num_samples):
    plt.subplot(221+i)
    plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
plt.show()

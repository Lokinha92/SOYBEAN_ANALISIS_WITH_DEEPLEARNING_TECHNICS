import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential #Classificador
from keras.layers import Dense, Dropout # Neuronio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

caminho = "./DADOS/soybean.csv"

def trantando_dominio(caminho_ds):

    ds = pd.read_csv(caminho_ds)

    for coluna in ds.columns:

        moda = ds[coluna].mode()[0]

        ds[coluna] = ds[coluna].replace('?', moda)

    return ds

soybean = trantando_dominio(caminho)

label = LabelEncoder()
classLabel = LabelEncoder()

# Separação de variáveis

atributos = soybean.iloc[:, 1:35].values
classe = soybean.iloc[:, 35].values

for i in range(34):
  atributos[:,i] = label.fit_transform(atributos[:,i])

# previnindo problema de autocorrelação excluindo a ultima coluna

atributos = atributos[:, 0:33]

# Label Encoding da classe

classe = classLabel.fit_transform(classe)

classe

# Divisao dos dados entre treino e teste

x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, classe, test_size=0.3, random_state=0)

# arrumando erro de tipo de entrada. aparentemente, o array com os dados de cada categoria deve ser convertido
# para o tipo de array desejado pela rede. um tensor_array.

x_treino = tf.convert_to_tensor(x_treino, dtype=tf.float32)
y_treino = tf.convert_to_tensor(y_treino, dtype=tf.float32)
x_teste = tf.convert_to_tensor(x_teste, dtype=tf.float32)
y_teste = tf.convert_to_tensor(y_teste, dtype=tf.float32)

# normalização dos dados com z-score

normalizador = StandardScaler()

x_treino = normalizador.fit_transform(x_treino)
x_teste = normalizador.fit_transform(x_teste)

# Dummy_encoding

y_treino = to_categorical(y_treino, 19)
y_teste = to_categorical(y_teste, 19)

#modelagem da rede -> 33 atributos de entrada, 2 camadas ocultas com 17 e 16 neuronios e 19 neuronios de saida

rede = Sequential()

rede.add(Dense(units=17, kernel_initializer= 'uniform', activation= 'relu', input_dim = 33)) # camada oculta com informação da entrada
rede.add(Dropout(0.2))
rede.add(Dense(units=16, kernel_initializer= 'uniform', activation= 'relu')) # camada oculta
rede.add(Dropout(0.2))

rede.add(Dense(units=19, kernel_initializer='uniform', activation='softmax')) # camada de saida

rede.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

treinamento = rede.fit(x_treino, y_treino, epochs=500, validation_data= (x_teste, y_teste))

#visualização das metricas de erro

treinamento.history.keys()
plt.plot(treinamento.history['val_loss']) # evolução do erro

plt.plot(treinamento.history['val_accuracy'])

previsoes = rede.predict(x_teste)

previsoes

# matriz de confusao

y_teste_matriz = [np.argmax(t) for t in y_teste]
previsoes_matriz = [np.argmax(t) for t in previsoes]

confusao = confusion_matrix(y_teste_matriz, previsoes_matriz)

plt.figure(figsize=(8, 6))
sns.set(font_scale=1.2)
sns.heatmap(confusao, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Classe Predita')
plt.ylabel('Classe Real')
plt.title('Matriz de Confusão')
plt.show()
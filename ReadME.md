<h1 align = 'center'>SOYBEAN ANALISIS WITH DEEP LEARNING TECHINCS</h1>

<strong><p align = center> GUSTAVO HENRIQUE D'ANUNCIAÇÃO FERREIRA</p></strong>

<h2 align = 'center'> ❓ Resumo </h2>

<b>Esta implementação consiste na análise e tratamento dos dados contidos no arquivo "soybean.csv" e da criação de um modelo de previsão utilizando técnicas de deep learning</b>

<h2 align = 'center'>🧩Apresentação da estrutura:</h2>

- As seguintes bibliotecas foram usadas durante a implementação:

```python
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
```

O pandas será utilizado para a leitura dos dados;

O numpy será usado para algumas operações uteis durante a análise;

Scikit-Learn será utilizado para codificação de categorias, normalização de dados, divisão dos dados em treino e teste e para a análise de metricas de erro;

O tensorflow e o Keras serão utilizados para a criação e configuração da rede neural artificial;

A matplotlib e o seaborn serão uteis para a visualização das metricas de erro e para a visualização da performance do modelo;

- Leitura e tratamento inicial de domínio dos dados

```python
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
```

Primeiro, o caminho referente à localização dos dados de entrada (soybean.csv) é referenciado e, caso a localização do arquivo seja modificada, a variável "caminho" deve ser atualizada com o endereço correto.

A função "tratando_dominio" recebe o caminho para o arquivo como entrada, realiza a leitura dos dados e faz o tratamento inicial dos dados inconsistentes com o domínio a cada atributo.
Esses dados fora do domínio estão representados como '?' dentro dos registros e, por se tratarem de variaveis categoricas, devem ser substituidos pelo valor da moda dentro do atributo em questão. A função passa por todos os atributos (colunas) do dataset, realiza esse tratamento dos dados e retorna o dataset já atualizado com os dados devidamente tratados.

Por fim, o objeto contendo o dataset é criado, juntamente com os objetos que serão usados para codificação dos atributos e da classe.

- Separação dos dados entre atributos e classe:

```python
atributos = soybean.iloc[:, 1:35].values
classe = soybean.iloc[:, 35].values
```

Como o atributo localizado originalmente na coluna 0 do dataset não tem valor de peso para a classificação, ele é removido durante a seleção de atributos.

- Codificação de atributos e prevenção de problemas de auto-correlação:

```python
#Label Encoding dos atributos
for i in range(34):
  atributos[:,i] = label.fit_transform(atributos[:,i])

# previnindo problema de autocorrelação excluindo a ultima coluna

atributos = atributos[:, 0:33]

# Label Encoding da classe

classe = classLabel.fit_transform(classe)

```

Devido ao fato de todas as instâncias do dataset se tratarem de dados categóricos, é necessário a aplicação do LabelEncoder em todas as colunas que compõem os atributos e tabmém para a classe.

O LabelEncoder irá associar um valor inteiro para cada registro único da coluna em questão.

- Divisão dos dados entre treino e teste

```python
x_treino, x_teste, y_treino, y_teste = train_test_split(atributos, classe, test_size=0.3, random_state=0)
```

Nessa divisão, os dados de treino representam 70% dos registros e os outros 30% ficam para os dados de teste.

- Tratamento dos dados de treino e teste para inserção correta no modelo

```python
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
```

Primeiramente, os dados de treino e teste são convertidos de numpy_array para tensor_array que é um formato de array reconhecido pelo modelo de RNA. Neste processo, o tipo de dado é convertido para float32 para que a normalização seja possível.

Com os arrays convertidos, os atributos e a classe são tratados de formas diferentes:

Para os atributos, é realizada uma normalização dos dados utilizando o z-score. A normalização dos dados pode contribuir para uma melhor performance do modelo.

Para a classe, é necessário que elas sejam passadas para o formato de dummy_class, que é representada por um array contendo 0's e 1's, onde a posição contendo o valor 1 representa a classe dos atributos em questão. Neste formato: [0., 0., 0., 1., 0., 0.,]

- Configuração, compilação e treinamento do modelo:

```python
#modelagem da rede -> 33 atributos de entrada, 2 camadas ocultas com 17 e 16 neuronios e 19 neuronios de saida

rede = Sequential()

rede.add(Dense(units=17, kernel_initializer= 'uniform', activation= 'relu', input_dim = 33)) # camada oculta com informação da entrada
rede.add(Dropout(0.2))
rede.add(Dense(units=16, kernel_initializer= 'uniform', activation= 'relu')) # camada oculta
rede.add(Dropout(0.2))

rede.add(Dense(units=19, kernel_initializer='uniform', activation='softmax')) # camada de saida

rede.compile(optimizer= 'adam', loss= 'categorical_crossentropy', metrics=['accuracy'])

treinamento = rede.fit(x_treino, y_treino, epochs=500, validation_data= (x_teste, y_teste))
```

Com todo o tratamento dos dados finalizado, o modelo pode finalmente ser configurado.

O modelo utilizado nessa implementação foi o Sequential.
O modelo de RNA Sequential é uma arquitetura de rede neural feedforward simples e linear que é comumente usada em tarefas de deep learning. É chamado de "sequencial" porque o modelo é criado camada por camada, em sequência.

A primeira camada oculta contém 17 neurônios, e também contém a informação dos parâmetro de entrada.
A segunda camada oculta contém 16 neurônios. Ambas as camadas utilizam a função de ativação 'relu'.
Após cada camada oculta, foi adicionado um Dropout de 20%, que serve para previnir o "overfitting" que é um problema de superajuste.

Para a camada de saída, 19 neuronios são adicionados, representando a quantidade de classes que devem ser previstas. A camada de saída utilizada é a "softmax" pois a resposta para a classe será dada em termos de probabilidade.

Depois de criada, a rede é compilada usando o optimizador "adam" que é um optimizador recomendado e a função de perda passada é a "categorical_crossentropy" pois se trata de um problema de classificação multiclasse. Utilizamos o "accuracy" como metrica de erro.

Por fim, o modelo é treinado através do método .fit passando os dados de treino como os 2 primeiros hiperparâmetros e o número de epochs (vezes que os dados são submetidos à RNA) pode ser alterado ajustando o hiperparâmetro "epochs". Como padrão, o número de epochs foi definido como 500, mas o modelo pode ser treinado com qualquer número de epochs.

Não é necessário criar um bloco de código com o calculo das métricas de erro, já que o modelo de RNA já faz esse calculo durante o treinamento, basta passar uma tupla contendo os dados de teste (respectivamente) para o hiperparâmetro "validation_data"

- Enfim, as métricas de erro e a performance do modelo podem ser visualizadas

```python
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
```

Para a visualização das métricas de erro, um gráfico é gerado:

<div align = center> <img align src = /img/erro.jpg> </div>

Em azul temos a "evolução" do erro e em laranja a evolução da performance do modelo. Pode-se notar que, como é o esperado, enquanto o valor do erro cai a cada epoch, a performance aumenta.

Para a visualização da performance do modelo, uma matriz de confusão é gerada:

<div align = center> <img align src = /img/confusao.jpg> </div>

Na diagonal principal, estão os dados classificados corretamente pelo modelo, e nas posições adjacentes da matriz os erros.

<h2 align = center>🔧 Compilação e execução </h2>

Para compilação e execução correta do código, a versão recomendada do Python é a 3.8.x ou superiores.

Também é importante ressaltar que as bibliotecas devem estar instaladas corretamente no ambiente de execução, caso contrário, erros vão ocorrer.
# Importação das bibliotecas necessárias
import pandas as pd  # Para manipulação e análise de dados
from sklearn.model_selection import train_test_split  # Para dividir os dados em treino e teste
from sklearn.preprocessing import MinMaxScaler  # Para normalizar os dados
from sklearn.feature_selection import RFE  # Para seleção recursiva de features (Recursive Feature Elimination)
from sklearn.linear_model import LogisticRegression  # Modelo de regressão logística
from sklearn.tree import DecisionTreeClassifier  # Modelo de Árvore de Decisão
from sklearn.ensemble import RandomForestClassifier  # Modelo de Floresta Aleatória (Random Forest)
from sklearn.svm import SVC  # Modelo de Máquina de Vetores de Suporte (SVM)
from sklearn.neighbors import KNeighborsClassifier  # Modelo de K-Nearest Neighbors (KNN)
from sklearn.metrics import accuracy_score  # Para calcular a acurácia das predições

# Importações para construir o modelo de rede neural com Keras
from tensorflow.keras.models import Sequential  # Para criar modelos sequenciais
from tensorflow.keras.layers import Dense, Dropout  # Camadas densas (fully connected) e Dropout para regularização
from tensorflow.keras.optimizers import Adam  # Otimizador Adam para treinamento da rede

# --------------------------------------------------------------------
# 1. Carregamento do Dataset
# --------------------------------------------------------------------
# Carrega o dataset de churn a partir de um arquivo CSV. 
# Esse dataset contém informações sobre clientes e se eles cancelaram (churn) ou não.
df = pd.read_csv('Churn.csv')

# --------------------------------------------------------------------
# 2. Pré-processamento dos Dados
# --------------------------------------------------------------------
# Trata a coluna 'Total Charges':
# - Substitui valores vazios ('') por 0
df['Total Charges'] = df['Total Charges'].replace('', 0)

# - Preenche valores nulos (NaN) com 0
df['Total Charges'] = df['Total Charges'].fillna(0)

# - Converte a coluna 'Total Charges' para tipo numérico (float), tratando erros e substituindo possíveis NaN por 0
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce').fillna(0)

# --------------------------------------------------------------------
# 3. Normalização dos Dados Numéricos
# --------------------------------------------------------------------
# Cria um objeto MinMaxScaler para normalizar os dados entre 0 e 1
scaler = MinMaxScaler()

# Normaliza as colunas 'tenure', 'Monthly Charges' e 'Total Charges'
df[['tenure', 'Monthly Charges', 'Total Charges']] = scaler.fit_transform(
    df[['tenure', 'Monthly Charges', 'Total Charges']]
)

# --------------------------------------------------------------------
# 4. Preparação dos Dados para Modelagem
# --------------------------------------------------------------------
# Separa as features (variáveis explicativas) e a variável alvo (target)
# - Remove as colunas 'Churn' (que será a variável alvo) e 'Customer ID' (não é útil para predição)
# - Converte variáveis categóricas em variáveis dummy (one-hot encoding)
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))

# Cria a variável alvo 'y' transformando 'Yes' em 1 e 'No' em 0
y = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# --------------------------------------------------------------------
# 5. Divisão dos Dados em Conjuntos de Treino e Teste
# --------------------------------------------------------------------
# Divide os dados em 80% para treinamento e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------------
# 6. Seleção de Features Utilizando RFE (Recursive Feature Elimination)
# --------------------------------------------------------------------
# Utiliza uma regressão logística como estimador base para o RFE
model_lr = LogisticRegression()

# Configura o RFE para selecionar as 10 melhores features com base na importância determinada pelo modelo
selector = RFE(model_lr, n_features_to_select=10)

# Ajusta (treina) o RFE com os dados de treinamento
selector = selector.fit(X_train, y_train)

# Identifica os nomes das colunas selecionadas pelo RFE
selected_columns = X_train.columns[selector.support_]
print("Colunas selecionadas:", selected_columns)

# Filtra os conjuntos de dados para manter apenas as features selecionadas
X_train_selected = X_train[selected_columns]
X_test_selected = X_test[selected_columns]

# --------------------------------------------------------------------
# 7. Função para Treinamento e Avaliação de Modelos Clássicos
# --------------------------------------------------------------------
def train_and_evaluate(model, model_name):
    """
    Treina um modelo de machine learning, realiza predições no conjunto de teste 
    e imprime a acurácia do modelo.
    
    Parâmetros:
        model : objeto do modelo a ser treinado (ex: LogisticRegression, DecisionTreeClassifier, etc.)
        model_name : nome do modelo (string) para exibição dos resultados
    """
    # Treina o modelo utilizando os dados de treinamento com features selecionadas
    model.fit(X_train_selected, y_train)
    
    # Realiza predições no conjunto de teste
    y_hat = model.predict(X_test_selected)
    
    # Calcula a acurácia comparando as predições com os valores reais do teste
    accuracy = accuracy_score(y_test, y_hat)
    
    # Imprime a acurácia formatada com 4 casas decimais
    print(f"Acurácia do modelo {model_name}: {accuracy:.4f}")
    
    return accuracy

# --------------------------------------------------------------------
# 8. Treinamento e Avaliação de Modelos Clássicos de Machine Learning
# --------------------------------------------------------------------
# Cria um dicionário contendo os modelos a serem testados
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Itera sobre cada modelo, treina e avalia, exibindo a acurácia de cada um
for name, model in models.items():
    train_and_evaluate(model, name)

# --------------------------------------------------------------------
# 9. Construção, Treinamento e Avaliação de um Modelo de Rede Neural com Keras
# --------------------------------------------------------------------
# Cria um modelo sequencial (empilhamento linear de camadas)
model_nn = Sequential()

# Adiciona a primeira camada oculta:
# - 64 neurônios
# - Função de ativação 'relu'
# - 'input_dim' definido com o número de features selecionadas
model_nn.add(Dense(units=64, activation='relu', input_dim=len(X_train_selected.columns)))

# Adiciona uma camada de Dropout para reduzir overfitting, descartando 50% dos neurônios aleatoriamente durante o treinamento
model_nn.add(Dropout(0.5))

# Adiciona uma segunda camada oculta com 128 neurônios e função de ativação 'relu'
model_nn.add(Dense(units=128, activation='relu'))

# Adiciona a camada de saída:
# - 1 neurônio, pois o problema é de classificação binária
# - Função de ativação 'sigmoid' para produzir uma saída entre 0 e 1 (probabilidade)
model_nn.add(Dense(units=1, activation='sigmoid'))

# Define o otimizador Adam com uma taxa de aprendizado de 0.001
optimizer = Adam(learning_rate=0.001)

# Compila o modelo especificando:
# - Função de perda: 'binary_crossentropy', apropriada para classificação binária
# - Otimizador: Adam
# - Métrica: acurácia (accuracy)
model_nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Treina o modelo neural:
# - epochs: número de iterações sobre o conjunto de dados
# - batch_size: número de amostras por atualização dos pesos
# - validation_data: dados de validação para monitorar o desempenho durante o treinamento
model_nn.fit(X_train_selected, y_train, epochs=200, batch_size=32, validation_data=(X_test_selected, y_test), verbose=1)

# Realiza predições no conjunto de teste usando o modelo neural
y_hat_nn = model_nn.predict(X_test_selected)

# Converte as predições (probabilidades) em classes:
# Se a probabilidade for menor que 0.5, atribui 0; caso contrário, atribui 1
y_hat_nn = [0 if val < 0.5 else 1 for val in y_hat_nn]

# Calcula e imprime a acurácia do modelo neural
print(f"Acurácia do modelo Neural: {accuracy_score(y_test, y_hat_nn):.4f}")

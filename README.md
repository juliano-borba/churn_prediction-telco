# Machine Learning Portfolio: Predição de Churn #

Este projeto demonstra a aplicação de técnicas de Machine Learning e Deep Learning para prever o churn (cancelamento de clientes). Nele, são exploradas diversas etapas essenciais, desde o pré-processamento dos dados até a avaliação de diferentes modelos de classificação.

## Descrição do Projeto
Os dados foram extraídos dessa base de dados do Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

O objetivo deste projeto é comparar a performance de diversos algoritmos de classificação para o problema de churn. O script realiza as seguintes etapas:


Carregamento e pré-processamento: Leitura do dataset, tratamento de valores nulos e normalização dos dados.

Preparação dos dados: Transformação de variáveis categóricas em variáveis dummy e separação da variável alvo.

Seleção de Features: Utilização do RFE (Recursive Feature Elimination) para selecionar as 10 melhores features.

Treinamento de Modelos Clássicos: Teste de algoritmos como Regressão Logística, Árvore de Decisão, Random Forest, SVM e KNN.

Modelo de Rede Neural: Construção, treinamento e avaliação de uma rede neural com Keras.

Comparação de Resultados: Impressão das acurácias de cada modelo para facilitar a análise de desempenho.

Estrutura do Repositório
churn_ml_script.py: Script principal com todo o código comentado e detalhado.

README.md: Este arquivo, com informações sobre o projeto, instruções e detalhes técnicos.

(Opcional) Outros arquivos ou pastas que contenham dados, resultados ou documentação adicional.

## Pré-requisitos

Para executar este projeto, você precisará ter instalado:

Python 3.x

Bibliotecas:

pandas
scikit-learn
tensorflow (inclui Keras)

## Detalhes do Script
O código está organizado nas seguintes seções:

Carregamento do Dataset:
Lê o arquivo CSV e carrega os dados em um DataFrame do pandas.

Pré-processamento:
Trata os valores ausentes e converte a coluna 'Total Charges' para o tipo numérico. Em seguida, normaliza as colunas numéricas utilizando MinMaxScaler.

Preparação dos Dados:
Converte variáveis categóricas em dummies e separa a variável alvo (Churn) transformando-a em valores binários (0 para "No" e 1 para "Yes").

Divisão dos Dados:
Divide o dataset em conjuntos de treino e teste (80%/20%).

Seleção de Features com RFE:
Utiliza uma regressão logística para selecionar as 10 melhores features do dataset.

Treinamento de Modelos Clássicos:
Treina e avalia modelos como Regressão Logística, Árvore de Decisão, Random Forest, SVM e KNN, exibindo a acurácia de cada um.

Modelo de Rede Neural com Keras:
Constrói uma rede neural com duas camadas ocultas e regularização via Dropout, treinando e avaliando sua performance no conjunto de teste.

Resultados
Após a execução do script, serão exibidas as acurácias dos diferentes modelos no terminal. Essa comparação ajuda a identificar qual abordagem se adapta melhor ao problema de churn neste dataset.

Contribuições
Contribuições são bem-vindas! Se você deseja melhorar este projeto, sinta-se à vontade para:

Abrir issues para reportar bugs ou sugerir melhorias.
Submeter pull requests com novas funcionalidades ou correções.
Licença
Este projeto está licenciado sob a MIT License.

Contato
Caso tenha dúvidas ou sugestões, entre em contato por meio das issues do GitHub ou pelo seu e-mail.

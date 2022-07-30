## Versão 01 - Voting
# serão criados e treinados 3 modelos (Regressão Logística, Random Forest e Multinomial NB)
# e os modelos passarão por uma votação ^^

import numpy as np

####################################################
# Run the Python interpreter and type the commands:
# >>> import nltk
# >>> nltk.download()
# https://www.nltk.org/data.html
####################################################

from nltk.corpus import stopwords

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score

# função load_files economiza o trabalho de loops para percorrer todos os arquivos.
# sklearn... seu lindo xD
news = load_files("data", encoding="utf-8", decode_error="replace")

# separando variáveis de entrada e saída
X = news.data  # dados
y = news.target  # labels

# definindo lista de palavras de parada
my_stop_words = set(stopwords.words("english"))

# Divisão de dados de treino e teste
TEST_SIZE = 0.3
RANDOM_STATE = 75

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# vetorização
MAX_FEATURES = 1000
vectorizer = TfidfVectorizer(
    norm=None,
    stop_words=my_stop_words,
    max_features=MAX_FEATURES,
    decode_error="ignore",
)
# treina o modelo e vetoriza os dados
X_treino_vectors = vectorizer.fit_transform(X_treino)
# vetoriza os dados para teste
X_teste_vectors = vectorizer.transform(X_teste)

## Criando os modelos
modelo1 = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", random_state=30, max_iter=1000
)
# mil árvore de decosões, até 100 folhas
modelo2 = RandomForestClassifier(n_estimators=1000, max_depth=100, random_state=1)

modelo3 = MultinomialNB()

result = []

## iniciando a votação
## recomendado usar número ímpar de modelos para a votação
voting_model = VotingClassifier(
    estimators=[("LG", modelo1), ("RF", modelo2), ("NB", modelo3)], voting="soft"
)
print(
    """
# ---------------------------------------------------------------------------- #
#                               Modelo de Votação                              #
# ---------------------------------------------------------------------------- #
      """
)
print(voting_model)

# train
voting_model = voting_model.fit(X_treino_vectors, y_teste)

# previsoes com dados de teste
previsoes = voting_model._predict(X_teste_vectors)

result.append(accuracy_score(y_teste, previsoes))

print(
    f"""
# ---------------------------------------------------------------------------- #
#                              Acurácia do modelo                              #
# ---------------------------------------------------------------------------- #
#                    {accuracy_score(y_teste, previsoes)}                      #
# {result}                                                                     #
# ---------------------------------------------------------------------------- #
      """
)

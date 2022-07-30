## Versão 01 - Voting
#serão criados e treinados 3 modelos (Regressão Logística, Random Forest e Multinomial NB)
# e os modelos passarão por uma votação ^^

import numpy as np
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import accuracy_score

# função load_files economiza o trabalho de loops para percorrer todos os arquivos.
# sklearn... seu lindo xD
news = load_files('data', encodiing = 'utf-8', decode_error = 'replace')

# separando variáveis de entrada e saída
X = news.data     # dados
y = news.target   # labels

# definindo lista de palavras de parada
my_stop_words = set(stopwords.words('english'))

# Divisão de dados de treino e teste
TEST_SIZE = .3
RANDOM_STATE = 75

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size = TEST_SIZE, random_state = RANDOM_STATE)

# vetorização
MAX_FEATURES = 1000
vectorizer = TfidfVectorizer(norm= None, stop_words=my_stop_words, max_features=MAX_FEATURES, decode_error='ignore')

X_treino_vectors = vectorizer.fit_transform(X_treino)
X_teste_vectors  = 4

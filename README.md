# Ensemble-learning-em-python

## Projetos

- [ ] Modelagem de tópicos do noticiário financeiro
  - Extrair, tratar e classificar textos para filtrar dados relevantes para auxílio de tomada de decisão do investidor

## Definição do projeto

> Com alguns parágrafos de texto, podemos afirmar sobre qual assunto é discutido?

Modelos de entrada: trechos de notícias
Modelos de saída: categorias, baseadas em dados históricos

> A etiquetagem é um processo demorado e CARO, geralmente bancos de dados etiquetados são guardados secretamente.

A aprendizagem ensemble é um paradigma de aprendizagem de máquina em que vários
modelos (frequentemente chamados de “estimadores fracos”) são treinados para resolver o
mesmo problema e combinados para obter melhores resultados. A hipótese principal é que
quando modelos fracos são combinados corretamente podemos obter modelos mais precisos
e/ou robustos.

## Conjuntos de dados

Consiste em 2.225 documentos do site de notícias da BBC, publicadas entre 2004 e 2005, correspondentes a histórias em cinco áreas temáticas:

1. negócios
2. entretenimento
3. política
4. esporte
5. tecnologia

Votin = todos os modelos fazem as previsões, e suas saídas passam por uma votação
Staking = as saídas dos modelos individuais alimentam um terceiro modelo

## Para saber mais

- https://web.engr.oregonstate.edu/~tgd/publications/mcs-ensembles.pdf
- https://youtu.be/dhvmVScjrzE

Machine Learning
Autor: Tom Michael
Machine Learning with Python for Everyone
Autor: Mark Fenne
The Hundred-Page Machine Learning Book
Autor: Andriy Burkov

D. Greene and P. Cunningham. ("Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering")[D. Greene and P. Cunningham. "Practical Solutions to the Problem of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 2006. ], Proc. ICML 2006. 

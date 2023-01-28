# %%
from pathlib import Path
import json
import pandas as pd

# %% [markdown]
# ## Data loading

# %%
news_path = Path('../cache/news_6771.json')
with news_path.open(encoding="UTF-8") as f:
    all_data = json.load(f).get('catalog')

# %% [markdown]
# Reducing number of categories

# %%
print(f"All categories: {set(pd.DataFrame(all_data).get('category'))}")

cats = ['culture', 'economics', 'gorod', 'politics']
news = [article for article in all_data if article.get('category') in cats]
print(f'Len of dataset: {len(news)}')

# %% [markdown]
# ## Corpus preprocessing

# %% [markdown]
# My corpus is big enough (60_000 < features) and M1 chip processed some models for 12+ hours.
# So, aim of so significant preprocessing is features redusing

# %%
import nltk
# nltk.download("stopwords")

from nltk.corpus import stopwords
from pymystem3 import Mystem
from string import punctuation
from nltk.stem.snowball import SnowballStemmer
import pymorphy2

morph = pymorphy2.MorphAnalyzer()

stemmer = SnowballStemmer("russian")
mystem = Mystem()

russian_stopwords = stopwords.words("russian")
english_stopwords = stopwords.words("english")

# %%
def preprocess_text(text):
    """
    Tokenize text and lemmatize it with pymystem3 and remove stopwords and punctuation symbols from it

    * Stemming is used for significant vocabulary reducing. In other case it hase dim > 60_000
    * Removed all punctuation simbols
    * Removed all numbers
    * Removed all small words
    * Removed site link (nn ru)
    * Removed unknown words
    """

    tokens = mystem.lemmatize(text.lower().replace('ё', 'е'))
    
    marks = '''
    !()-[]{};?@#$%:'"\,./^&amp;*_—«»–…’‘“”
    '''
 
    new_tokens = []
    for token in tokens:
        token = stemmer.stem(token)
        if len(token) < 2: continue
        for x in token:
            if x in marks:
                token = token.replace(x, "")
        new_tokens.append(token)
    tokens = new_tokens


    blacklist = ['\n', '\t', '\r', ' ', '', *punctuation, *russian_stopwords, *english_stopwords, 'nn', 'ru']
    tokens = [token for token in tokens if token not in blacklist and not token.isdigit() and morph.word_is_known(token)]

    text = " ".join(tokens)
    return text

# %%
from sklearn.feature_extraction.text import CountVectorizer
from typing import Optional

class CorpusStructure:
    """
    Structure of corpus
    """
    corpus: list
    target: list
    vectorizer: CountVectorizer
    matrix: Optional[list] = None

    def __init__(self, corpus: list, targets: list, vectorizer: CountVectorizer) -> None:
        self._corpus = corpus
        self._target = targets
        self._vectorizer = vectorizer
        self._matrix = None

    @property
    def corpus(self) -> list:
        return self._corpus

    @property
    def target(self) -> list:
        return self._target

    @property
    def corpus_len(self) -> int:
        return len(self._corpus)

    @property
    def target_len(self) -> int:
        return len(self._target)

    @property
    def matrix(self):
        if self._matrix == None:
            self._matrix = self._vectorizer.transform(self._corpus)
        return self._matrix.toarray()

    def transform(self):
        return self._vectorizer.transform(self._corpus)

# %%
from sklearn.model_selection import train_test_split
    
def train_test_partition(corpus: list, targets: list, vectorizer: CountVectorizer, test_size: float = 0.2) -> tuple:
    """
    Function that creates train and test partition
    """
    X_train, X_test, y_train, y_test = train_test_split(corpus, targets, test_size=test_size)
    return CorpusStructure(X_train, y_train, vectorizer), CorpusStructure(X_test, y_test, vectorizer)

# %%
import random

def get_corpus(news: list, n: int = -1, shuffle=True):
    """
    Provides preprocessing for corpus
    """
    corpus_text = []
    corpus_target = []
    n = len(news) if n == -1 else n
    if shuffle: random.shuffle(news)
    for article in news[:n]:
        corpus_text.append(preprocess_text(article.get('text')))
        corpus_target.append(article.get('category'))
    return corpus_text, corpus_target

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

train_path = Path('../cache/vectorizer_train.pkl')
test_path = Path('../cache/vectorizer_test.pkl')

if train_path.exists() and test_path.exists():
    train_data = pickle.load(open('../cache/vectorizer_train.pkl', 'rb'))
    test_data = pickle.load(open('../cache/vectorizer_test.pkl', 'rb'))
else:
    vectorizer = TfidfVectorizer()
    corpus_text, corpus_target = get_corpus(news)
    vectorizer.fit(corpus_text)

    train_data, test_data = train_test_partition(corpus_text, corpus_target, vectorizer, test_size=0.2)

    pickle.dump(train_data, open('../cache/vectorizer_train.pkl', 'wb'))
    pickle.dump(test_data, open('../cache/vectorizer_test.pkl', 'wb'))

# %% [markdown]
# Without stemmer: 35_053 \
# Processing time: > 12h 
# 
# With stemmer: 14_566\
# Processing time: 0.5h

# %%
len(vectorizer.get_feature_names_out())

# %% [markdown]
# > <b>Note</b> \
# Removing categories ['world', 'incidents'] did not reduce features len. \
# So, I suppose that the my site could use common language style for all news. It will lead to model accuracy
# 

# %% [markdown]
# ## EXPERIMENTS

# %% [markdown]
# To determine the best parameters for models I will use Grid Search

# %%
from sklearn import metrics
from sklearn.model_selection import GridSearchCV

"""Module for grid search and classification of the data"""
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


models = [
    (
        'LogisticRegression',
        LogisticRegression(),
        {
            'solver'    : ['newton-cg'],
            'max_iter'  : [1000]
        }
    ),
    (
        'MultinomialNB',
        MultinomialNB(),
        {
            'alpha': [0.1]
        }
    ),
    (
        'LinearSVC',
        LinearSVC(),
        {
            'loss'      : ['hinge'],
            'max_iter'  : [1000]
        }
    ),
    # (
    #     'SGDClassifier',
    #     SGDClassifier(),
    #     {
    #         'penalty'       : ['l1','l2'],
    #         'alpha'         : [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000],
    #         'learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
    #         'max_iter'      : [100],
    #         'loss'          : ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'],
    #     }
    # ),
    (
        'RandomForestClassifier',
        RandomForestClassifier(),
        {
            'criterion' : ['gini']
        }
    ),
    (
        'KNeighborsClassifier',
        KNeighborsClassifier(),
        {
            'weights'    : ['distance'],
            'n_neighbors': [9, 10, 11],
            'p'          : [2]
        }
    ),
    (
        'DecisionTreeClassifier',
        DecisionTreeClassifier(),
        {
            'criterion'     : ['gini'],
            'max_features'  : ['sqrt']
        }
    )
]

# %%
def plot_confusion_matrix(name: str, test: list, predicted: list, target: list):
    disp = metrics.ConfusionMatrixDisplay(metrics.confusion_matrix(test, predicted, labels=target))
    disp.plot()
    disp.ax_.set_title(f'{name} Confusion Matrix')

# %%
import matplotlib.pyplot as plt
import pandas as pd
import time
import pickle

cache_path = Path('../cache')

accuracy = []
precision = []
recall = []
f1 = []

for i, (name, model, params) in enumerate(models):
    start_time = time.perf_counter()
    grid_classifier = GridSearchCV(model,
                                    params,
                                    cv=10,
                                    scoring='accuracy',
                                    verbose=0,
                                    error_score=0,
                                    n_jobs=-1
                                    )
    grid_classifier.fit(train_data.matrix, train_data.target)
    predicted = grid_classifier.predict(test_data.matrix)

    accuracy.append(metrics.accuracy_score(test_data.target, predicted))
    precision.append(metrics.precision_score(test_data.target, predicted, average='micro', zero_division=0))
    recall.append(metrics.recall_score(test_data.target, predicted, average='micro', zero_division=0))
    f1.append(metrics.f1_score(test_data.target, predicted, average='micro', zero_division=0))

    print(f'Finished {name} in {time.perf_counter() - start_time:.2f} seconds. Accuracy: {accuracy[-1]}', flush=True)
    print(f'Best parameters: {grid_classifier.best_params_}')
    
    # ckpt_name = cache_path / f'{name}_{accuracy}.pkl'
    # names = list(cache_path.glob(f'{name}_*.pkl'))
    # if len(names) and accuracy > names[-1].split('_')[-1]:
    #     with ckpt_name.open() as file_descr:
    #         pickle.dump(grid_classifier, file_descr)

    plot_confusion_matrix(name, test_data.target, predicted, corpus_target)

metrics_frame = pd.DataFrame({
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}, index=[name for name, _, _ in models])

metrics_frame

# %%
"""
Finished LogisticRegression in 139.23 seconds. Accuracy: 0.8350515463917526
Best parameters: {'max_iter': 1000, 'solver': 'newton-cg'}


Finished MultinomialNB in 15.37 seconds. Accuracy: 0.7961053837342497
Best parameters: {'alpha': 0.1}
Finished RandomForestClassifier in 41.84 seconds. Accuracy: 0.8006872852233677
Best parameters: {'criterion': 'gini'}
Finished KNeighborsClassifier in 245.95 seconds. Accuracy: 0.7789232531500573
Best parameters: {'n_neighbors': 10, 'p': 2, 'weights': 'distance'}
Finished DecisionTreeClassifier in 10.31 seconds. Accuracy: 0.5715922107674685
Best parameters: {'criterion': 'gini', 'max_features': 'sqrt'}
""";

# %%




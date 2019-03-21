#%%

"""
# Problem set 1
"""
#%%

import spacy

nlp = spacy.load("en", disable=["ner"])

#%%

from zipfile import ZipFile
import re
from tqdm import tqdm
import random

with ZipFile("cases.zip") as zipfile:

    ids = []
    year = {}
    cases = {}
    for name in sorted(zipfile.namelist()):
        m = re.match(r"(\d{4})_(\w*).txt", name)
        id = m.group(2)
        ids.append(id)
        year[id] = int(m.group(1))
        cases[id] = zipfile.open(name).read().decode("utf8")


#%%

ids = random.choices(ids, k=500)
cases = [cases[id] for id in ids]
year = [year[id] for id in ids]
docs = list(tqdm(nlp.pipe(cases, batch_size=10)))

#%%

import matplotlib.pyplot as plt

plt.title("Number of sentences")
plt.hist([sum(1 for _ in doc.sents) for doc in docs])

#%%
plt.title("Number of words")
plt.hist([sum(1 for _ in doc) for doc in docs])

#%%
plt.title("Number of letters")
plt.hist([sum(1 for _ in doc.text) for doc in docs])

#%%
plt.title("Number of nouns")
plt.hist([sum(tok.pos_ == "NOUN" for tok in doc) for doc in docs])

#%%
plt.title("Number of verbs")
plt.hist([sum(tok.pos_ == "VERB" for tok in doc) for doc in docs])

#%%
plt.title("Number of adjectives")
plt.hist([sum(tok.pos_ == "ADJ" for tok in doc) for doc in docs])

#%%
from operator import itemgetter


def normalize(doc):
    ans = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space:
            continue
        norm = tok.lemma_
        if norm:
            ans.append((norm, tok.pos_))
    return ans


def make_trigrams(words):
    return list(zip(words, words[1:], words[2:]))


def filter_for_nouns(trigrams):
    return [
        tuple(map(itemgetter(0), tri))
        for tok, tri in zip(doc[2:], trigrams)
        if tri[-1][1] == "NOUN"
    ]


trigrams = [filter_for_nouns(make_trigrams(normalize(doc))) for doc in docs]

#%%
from collections import Counter
import itertools
from functools import partial

most_common_trigrams = [
    t for t, _ in Counter(itertools.chain.from_iterable(trigrams)).most_common(1000)
]

#%%
def make_feature(most_common_trigrams, trigrams):
    c = Counter(trigrams)
    return list(map(c.__getitem__, most_common_trigrams))


#%%

import pandas as pd
import numpy as np

X = np.array(list(list(map(partial(make_feature, most_common_trigrams), trigrams))))
X = X / X.std(axis=0)

#%%

df = pd.DataFrame({"caseid": ids, **dict(zip(most_common_trigrams, X.T))}).merge(
    pd.read_csv("case_reversed.csv"), on="caseid"
)
y = df.case_reversed.values

#%%
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(solver="liblinear")

model.fit(X_train, y_train)

print(f"Training accuracy {model.score(X_train, y_train)}")
print(f"Training F1 {model.score(X_train, y_train)}")
print(f"Test accuracy {model.score(X_test, y_test)}")
print(f"Test F1 {model.score(X_test, y_test)}")


#%%

model = GridSearchCV(
    LogisticRegression(solver="liblinear"),
    {"penalty": ("l1", "l2"), "C": [10 ** i for i in range(-5, 5)]},
    cv=5,
    iid=False,
)
model.fit(X, y)

print(f"Best model got {model.best_score_} accuravy with params {model.best_params_}")

#%%
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

scores = cross_val_predict(model.best_estimator_, X, y, cv=5)
fpr, tpr, thresholds = roc_curve(y, scores)

plt.plot(
    fpr,
    tpr,
    color="darkorange",
    label=f"ROC curve (area = {roc_auc_score(y, scores):.02f})",
)
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()


#%%
from operator import attrgetter

sentences = list(
    map(
        attrgetter("text"),
        itertools.chain.from_iterable(map(attrgetter("sents"), docs)),
    )
)

sentences = random.choices(sentences, k=10000)
#%%
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

scores = list(map(itemgetter("compound"), map(sid.polarity_scores, sentences)))

lowest = np.argsort(scores)[:10]
print("lowest compound scores:")
print(*map(sentences.__getitem__, lowest), sep="\n")
print("highest compound scores:")
highest = np.argsort(scores)[-10:]
print(*map(sentences.__getitem__, highest), sep="\n")

#%%
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(sentences)

#%%
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(X)

aux = np.unravel_index(np.argsort(similarities.ravel()), similarities.shape)

#%%

print("Dissimilar sentences")
for a, b in aux[:5]:
    print(sentences[a])
    print(sentences[b])
    print()

print("Similar sentences")
for a, b in aux[-5:]:
    print(sentences[a])
    print(sentences[b])
    print()

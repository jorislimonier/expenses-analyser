# %%
# Third-party imports
from sklearn.naive_bayes import MultinomialNB
from itertools import count
from sklearn.feature_extraction.text import TfidfTransformer,  TfidfVectorizer, CountVectorizer
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
import plotly.express as px
import numpy as np
import pandas as pd
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

# Custom imports
import data_loader
import classify

# %%


def reload():
    importlib.reload(data_loader)
    importlib.reload(classify)


# %% [markdown]
# ## Decision Tree
# %%
reload()
dl = data_loader.DataLoader()
cls = classify.Classify(dl)
cls.apply_decision_tree()


# %% [markdown]
# ## Random Forest

# %%
reload()
dl = data_loader.DataLoader()
cls = classify.Classify(dl)
cls.apply_random_forest(True)

# %% [markdown]
# ## Load JSON

# %%

import json

with open("label_identifiers.json") as f:
    label_id = json.load(f)
type(label_id)


















# %%
dl.debit["label"].value_counts()
dl.debit[dl.debit["label"] == "taxes_and_utilities"]
# %% [markdown]
# ## NLP on communication column

# %%

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dl.debit["communication"].dropna())
count_vect.vocabulary_
# %%

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf
# %%
mnb = MultinomialNB().fit(
    X_train_tfidf, dl.debit["label"][~dl.debit["communication"].isna()])

X_new_counts = count_vect.transform(["sfr phone bill", "", "carrefour market"])
X_new_tfidf = tfidf_transformer.transform(X_new_counts)
X_new_tfidf
mnb.predict(X_new_tfidf)

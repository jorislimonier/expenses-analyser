# %%
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
# ## NLP on communication column

# %%
for comm in dl.debit["communication"]:
    print(comm)
# %%

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(dl.debit["communication"].dropna())
count_vect.vocabulary_
# %%
tf_transformer = TfidfTransformer(use_idf=False)
X_train_tf = tf_transformer.fit_transform(X_train_counts)
print(X_train_tf)
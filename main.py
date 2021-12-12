# %%
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
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# %%
y_test.idxmax(axis=1)
# %%
y_pred_label = pd.DataFrame(y_pred).idxmax(
    axis=1).apply(lambda x: label_col[x])
# %%
skplt.metrics.plot_confusion_matrix(y_test.idxmax(axis=1),
                                    pd.DataFrame(y_pred).idxmax(axis=1).apply(lambda x: label_col[x]), x_tick_rotation=90)
# %%
y.idxmax(axis=1)[y.idxmax(axis=1) == "label_tuition_fees"]

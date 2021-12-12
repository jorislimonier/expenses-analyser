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
import data_loader
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
# %%

importlib.reload(data_loader)

dl = data_loader.DataLoader()
dl.label_encode_debit()
dl.split_date()

# %%

X = dl.debit.drop(columns=["label", "communication"])
y = dl.debit["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=.8,
                                                    shuffle=True,
                                                    random_state=42)

# %% [markdown]
# ## Decision Tree
# %%

dl = data_loader.DataLoader()
dl.split_date()

# %%

X = dl.debit.drop(columns=["label", "communication"])
y = dl.debit["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=.8,
                                                    shuffle=True,
                                                    random_state=42)
# %%
df_dummy = pd.get_dummies(
    dl.debit, columns=["transfer_type", "label"], drop_first=True)

label_col = [col for col in df_dummy.columns if col.startswith("label")]
y = df_dummy[label_col]
X = df_dummy.drop(columns=label_col+["communication"])
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=.8,
                                                    shuffle=True,
                                                    random_state=42)
clf = DecisionTreeClassifier(random_state=4)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

plt.figure(figsize=(64, 24))
plot_tree(clf, fontsize=6)
plt.savefig("tree", dpi=100)

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
y_pred_label = pd.DataFrame(y_pred).idxmax(axis=1).apply(lambda x: label_col[x])
# %%
skplt.metrics.plot_confusion_matrix(y_test.idxmax(axis=1), pd.DataFrame(
    y_pred).idxmax(axis=1).apply(lambda x: label_col[x]), x_tick_rotation=90
)
# %%
y.idxmax(axis=1)[y.idxmax(axis=1) == "label_tuition_fees"]
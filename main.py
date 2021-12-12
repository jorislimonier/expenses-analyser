# %%
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import plotly.express as px
import numpy as np
import pandas as pd
import importlib
import data_loader
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

# %%
clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)
# %%
X_test.join(pd.DataFrame({"pred": y_pred}))
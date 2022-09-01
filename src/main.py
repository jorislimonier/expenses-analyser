# %%
import importlib
import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from IPython.display import display
from scipy.spatial.distance import cosine
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    plot_confusion_matrix,
)

import data_loader
import labels_generator

importlib.reload(data_loader)
importlib.reload(labels_generator)


#%%
generator = labels_generator.LabelsGenerator()
# %%
generator.tokenize()
generator.embed()
# %%
generator.reduce_dim()
generator.plot_clusters()

#%%
df = generator.dim_red

df
#%%
dl = data_loader.DataLoader(PATH="../data/raw/expenses_new.csv")
dl.debit
# df
# %%
df_merged = pd.merge(
    left=dl.debit,
    right=generator.dim_red[["label_predicted"]],
    left_index=True,
    right_index=True,
    validate="1:1",
)
for lab in df_merged["label_predicted"].unique().sort_values():
    print(f"------ {lab} ------")
    display(df_merged[df_merged["label_predicted"] == lab])

# %%

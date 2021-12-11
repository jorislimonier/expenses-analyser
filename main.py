# %%
import plotly.express as px
import numpy as np
import pandas as pd
import importlib
import data_loader
# %%

importlib.reload(data_loader)
dl = data_loader.DataLoader()

dl.debit["label"].value_counts(dropna=False)

# %%


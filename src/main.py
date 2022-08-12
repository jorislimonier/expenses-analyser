# %%
import importlib
import json

import hdbscan
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
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

import data_loader

#%%
importlib.reload(data_loader)


class LabelsGenerator:
    """Label data"""

    def __init__(
        self,
        dl=data_loader.DataLoader(PATH="../data/expenses_new.csv"),
        pretrained_model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased",
    ) -> None:
        self.dl = dl
        self.debit = self.dl.debit.drop(columns=["label"])
        self.pretrained_model_name = pretrained_model_name

    def tokenize(self) -> None:
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        texts = self.debit["communication"].tolist()
        self.tokens = tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

    def embed(self) -> None:
        model = AutoModel.from_pretrained(self.pretrained_model_name)
        # Get the embeddings
        with torch.no_grad():
            self.embeddings = model(
                **self.tokens, output_hidden_states=True, return_dict=True
            ).pooler_output

    def reduce_dim(self) -> None:
        umap_coord = UMAP(init="spectral").fit_transform(self.embeddings)

        self.dim_red = pd.DataFrame(
            data=umap_coord, columns=[f"coord_{i}" for i in range(umap_coord.shape[1])]
        )
        display(self.dim_red)

        self.clusterer = hdbscan.HDBSCAN()
        self.clusterer.fit(umap_coord)
        labels = self.clusterer.labels_
        labels = [str(label) for label in labels]

        self.dim_red["label"] = labels

        print(self.dim_red.shape)

    def plot_clusters(self) -> go.Figure:
        """Plot the clusters"""

        if not hasattr(self, "tokens"):
            print("---tokenizing")
            self.tokenize()

        if not hasattr(self, "embeddings"):
            print("---embedding")
            self.embed()

        if not hasattr(self, "dim_red"):
            print("---reducing dimensions")
            self.reduce_dim()

        fig = px.scatter(
            data_frame=self.dim_red, x="coord_0", y="coord_1", color="label"
        )

        return fig


generator = LabelsGenerator()

#%%
generator.plot_clusters()

# %%
pd.DataFrame(data=generator.dim_red, columns=["a", "b"])

import hdbscan
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from transformers import AutoModel, AutoTokenizer
from umap import UMAP

import data_loader


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
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    def embed(self) -> None:
        """Create embeddings with the bert model."""
        model = AutoModel.from_pretrained(self.pretrained_model_name)

        # Get the embeddings
        with torch.no_grad():
            self.embeddings = model(
                **self.tokens,
                output_hidden_states=True,
                return_dict=True,
            ).pooler_output

    def reduce_dim(self) -> None:
        # Add `, y=self.dl.debit["label"]` with encoded labels later
        umap_coord = UMAP().fit_transform(self.embeddings)

        self.dim_red = pd.DataFrame(
            data=umap_coord,
            columns=[f"coord_{i}" for i in range(umap_coord.shape[1])],
        )
        display(self.dim_red)

        self.clusterer = hdbscan.HDBSCAN()
        self.clusterer.fit(umap_coord)
        self.dim_red["label_predicted"] = pd.Categorical(self.clusterer.labels_)
        print(self.dim_red[["label_predicted"]].dtypes)

        print(self.dim_red.shape)

    def plot_clusters(self) -> go.Figure:
        """Plot the clusters from vectorization + dimensionality reduction"""

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
            data_frame=self.dim_red.sort_values("label_predicted"),
            x="coord_0",
            y="coord_1",
            color="label_predicted",
        )

        return fig

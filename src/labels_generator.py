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
        dl=data_loader.DataLoader(PATH="../data/raw/expenses_new.csv"),
        pretrained_model_name: str = "princeton-nlp/sup-simcse-bert-base-uncased",
    ) -> None:
        self.dl = dl
        self.debit = self.dl.debit.drop(columns=["label"])
        self.pretrained_model_name = pretrained_model_name

    def tokenize(self) -> None:

        print("---tokenizing")
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

        print("---embedding")
        model = AutoModel.from_pretrained(self.pretrained_model_name)

        # Get the embeddings
        with torch.no_grad():
            self.embeddings = model(
                **self.tokens,
                output_hidden_states=True,
                return_dict=True,
            ).pooler_output

        print(self.embeddings)

    def reduce_dim(self, nb_dim=5) -> None:
        """Perform dimensionality reduction on the embeddings."""

        print("---reducing dimensions")
        # Add `, y=self.dl.debit["label"]` with encoded labels later
        manual_labels = np.repeat(a=[-1], repeats=self.embeddings.shape[0])
        


        mapper = UMAP(n_components=nb_dim).fit_transform(
            X=self.embeddings,
            # y=manual_labels,
            y=self.dl.debit["label"],
        )

        self.dim_red = pd.DataFrame(
            data=mapper,
            columns=[f"coord_{i}" for i in range(mapper.shape[1])],
        )
        display(self.dim_red)

        self.clusterer = hdbscan.HDBSCAN()
        self.clusterer.fit(mapper)
        self.dim_red["label_predicted"] = pd.Categorical(self.clusterer.labels_)
        print(self.dim_red[["label_predicted"]].dtypes)

        print(self.dim_red.shape)

    def plot_clusters(self) -> go.Figure:
        """Plot the clusters after performing:
        - Tokenization
        - Embedding
        - Dimensionality Reduction
        """

        if not hasattr(self, "tokens"):
            self.tokenize()

        if not hasattr(self, "embeddings"):
            self.embed()

        if not hasattr(self, "dim_red"):
            self.reduce_dim()

        fig = px.scatter_matrix(
            data_frame=self.dim_red.sort_values("label_predicted"),
            # x="coord_0",
            # y="coord_1",
            dimensions=[col for col in self.dim_red if col != "label_predicted"],
            color="label_predicted",
            opacity=0.7,
        )

        return fig

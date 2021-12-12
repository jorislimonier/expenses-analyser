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
from data_loader import DataLoader


class Classify():
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def apply_decision_tree(self, plot=False):
        """
        Apply a decision tree to the `debit` dataframe
        from the `DataLoader` class
        """
        label_col = [
            col for col in self.data_loader.debit_dummy.columns if col.startswith("label")]
        # ignore communication column for now
        X = self.data_loader.debit_dummy.drop(
            columns=label_col+["communication"])
        y = self.data_loader.debit_dummy[label_col]

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=.8,
                                                            shuffle=True,)
        # classify
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Decision tree accuracy: {accuracy_score(y_test, y_pred)}")

        if plot:
            plt.figure(figsize=(120, 60))
            plot_tree(clf, feature_names=X.columns)
            plt.savefig("tree", dpi=80)

    def apply_random_forest(self, conf_max=False):
        """
        Apply a decision tree to the `debit` dataframe
        from the `DataLoader` class
        """

        label_col = [
            col for col in self.data_loader.debit_dummy.columns if col.startswith("label")]

        # ignore communication column for now
        X = self.data_loader.debit_dummy.drop(
            columns=label_col+["communication"])
        y = self.data_loader.debit_dummy[label_col]

        # split data
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            train_size=.8,
                                                            shuffle=True,)
        # classify
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(f"Decision tree accuracy: {accuracy_score(y_test, y_pred)}")

        if conf_max:
            y_test_label = y_test.idxmax(axis=1)
            y_pred_label = pd.DataFrame(y_pred).idxmax(axis=1)
            y_pred_label = y_pred_label.apply(lambda x: label_col[x])
            skplt.metrics.plot_confusion_matrix(y_true=y_test_label,
                                                y_pred=y_pred_label,
                                                labels=label_col,
                                                x_tick_rotation=90)

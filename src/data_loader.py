import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    def __init__(self, PATH="expenses.csv") -> None:
        self.PATH = PATH
        self.load_data()
        self.generate_debit_df()
        self._debit_dummy = None

    def load_data(self):
        "load initial data"
        self.initial_data = pd.read_csv(self.PATH, header=3, sep=";")
        data = self.initial_data.copy()
        new_col_dict = {
            "Date d'opération": "date",
            "Type de mouvement": "transfer_type",
            "Montant": "amount",
            "Devise": "currency",
            "Communication": "communication",
        }
        data = data.rename(columns=new_col_dict)
        data["amount"] = data["amount"].str.replace(",", ".")
        data["amount"] = data["amount"].astype(float)
        data = data.convert_dtypes()
        data["date"] = pd.to_datetime(data["date"], format=r"%d/%m/%Y")
        self.data = data

    def make_initial_categories(self, comm):
        "categorize expenses based on communication"
        try:
            comm_lower = comm.lower()
        except Exception:
            return "taxes_and_utilities"

        with open("../data/label_identifiers.json") as f:
            cat_dict_json = json.load(f)
        for category, category_list in cat_dict_json.items():
            for marker in category_list:
                if marker in comm_lower:
                    return category
        return np.nan

    def generate_debit_df(self):
        "Generate a dataframe with debits only"

        # Create dataframe from self.data
        debit: pd.DataFrame = self.data.copy()
        debit = debit[
            np.all(
                [debit["amount"] < 0, debit["transfer_type"] != "DECOMPTE VISA"], axis=0
            )
        ]
        debit = debit.reset_index(drop=True)
        debit["spent"] = debit.amount * -1

        # drop currency column if all debits are euros and drop amount column
        assert np.all(
            debit["currency"].unique() == ["EUR"]
        ), "Some currencies are different from 'EUR'"

        debit.drop(columns=["amount", "currency"], inplace=True, errors="ignore")

        # Fill na with empty string
        debit["communication"] = debit["communication"].fillna("")

        # Make labels
        debit["label"] = debit["communication"].apply(self.make_initial_categories)

        self.debit = debit

    @property
    def debit_dummy(self):
        if self._debit_dummy is None:
            debit = DataLoader.split_date(self.debit)
            self._debit_dummy = pd.get_dummies(
                debit, columns=["transfer_type", "label"], drop_first=False
            )
        return self._debit_dummy

    def label_encode_debit(self):
        transfer_type_enc = LabelEncoder().fit_transform(self.debit["transfer_type"])
        label_enc = LabelEncoder().fit(self.debit["label"])
        self.debit["label"] = label_enc.transform(self.debit["label"])
        self.debit["transfer_type"] = transfer_type_enc

    @staticmethod
    def split_date(df):
        """
        Transform the `date` column of `df`
        from datetime object to year, month, day and dayofweek
        """
        df["year"] = pd.DatetimeIndex(df["date"]).year
        df["month"] = pd.DatetimeIndex(df["date"]).month
        df["day"] = pd.DatetimeIndex(df["date"]).day
        df["dayofweek"] = pd.DatetimeIndex(df["date"]).dayofweek
        df = df.drop(columns=["date"])
        return df

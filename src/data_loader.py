import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataLoader:
    def __init__(self, PATH="expenses.csv") -> None:
        self.PATH = PATH
        self.load_data()
        self.generate_debit_df()

    def load_data(self):
        """
        Load initial data and convert types as they should be.
        """
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

        with open("../data/external/label_identifiers.json") as f:
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
        filters = [debit["amount"] < 0, debit["transfer_type"] != "DECOMPTE VISA"]
        debit = debit[np.all(a=filters, axis=0)]
        debit = debit.reset_index(drop=True)
        debit["spent"] = debit.amount * -1

        # Drop currency column if all debits are euros and drop amount column
        all_eur = np.all(debit["currency"].unique() == ["EUR"])
        assert all_eur, "Some currencies are different from 'EUR'"

        debit.drop(columns=["amount", "currency"], inplace=True, errors="ignore")

        # Fill na with empty string
        debit["communication"].fillna(value="", inplace=True)

        # Make labels
        debit["label"] = debit["communication"].apply(self.make_initial_categories)

        self.debit = debit

    @staticmethod
    def split_date(df):
        """
        Transform the `date` column of `df`
        from datetime object to year, month, day and dayofweek
        """
        df["year"] = df["date"].apply(lambda x: x.year)
        df["month"] = df["date"].apply(lambda x: x.month)
        df["day"] = df["date"].apply(lambda x: x.day)
        df["dayofweek"] = df["date"].apply(lambda x: x.dayofweek)
        df = df.drop(columns=["date"])
        return df

    def label_encode_debit(self):
        ohe = OneHotEncoder()
        self.debit["label"] = ohe.fit_transform(self.debit["label"])

        ohe = OneHotEncoder()
        self.debit["transfer_type"] = ohe.fit_transform(self.debit["transfer_type"])

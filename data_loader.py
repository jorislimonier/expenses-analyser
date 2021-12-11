import plotly.express as px
import numpy as np
import pandas as pd


class DataLoader():
    def __init__(self, PATH="expenses.csv") -> None:
        self.PATH = PATH
        self.load_data()
        self.generate_debit_df()

    def load_data(self):
        "load initial data"
        self.initial_data = pd.read_csv(self.PATH, header=3, sep=";")
        data = self.initial_data.copy()
        data["Montant"] = data["Montant"].str.replace(",", ".")
        data["Montant"] = data["Montant"].astype(float)
        data = data.convert_dtypes()
        data["Date d\'opération"] = pd.to_datetime(data["Date d\'opération"])
        self.data = data

    def make_initial_categories(self, comm):
        "categorize expenses based on communication"
        try:
            comm_lower = comm.lower()
        except Exception as e:
            return np.nan

        if "escota vinci" in comm_lower:
            return "tolls"
        elif "aprr" in comm_lower:
            return "tolls"

        elif "helios crou 1 sc2341529" in comm_lower:
            return "crous_food"

        elif "pharmacie" in comm_lower:
            return "health"

        elif "maxicoffee sud" in comm_lower:
            return "food_non_groceries"
        elif "pizza" in comm_lower:
            return "food_non_groceries"
        elif "pizzeria" in comm_lower:
            return "food_non_groceries"
        elif "mirage sarl 4173455" in comm_lower:  # hop store (bar)
            return "food_non_groceries"
        elif "rpa sta acro3" in comm_lower:  # beer with generationZ
            return "food_non_groceries"
        elif "colgan, antibes" in comm_lower:  # The Duke bar
            return "food_non_groceries"

        elif "carrefourmarket, mougins" in comm_lower:
            return "groceries"
        elif "leader, antibes" in comm_lower:
            return "groceries"
        elif "sm casino cs898, biot" in comm_lower:
            return "groceries"
        elif "carrefour niceli" in comm_lower:
            return "groceries"
        elif "sc baghera, 06mougins" in comm_lower:
            return "groceries"
        elif "sm casino cs638" in comm_lower:
            return "groceries"

        elif "boulanger, mandelieu" in comm_lower:
            return "tech"
        elif "darty" in comm_lower:
            return "tech"

        elif "retrait bancomat" in comm_lower:
            return "cash"

        elif "esso moulieres" in comm_lower:
            return "gas"
        elif "q8 martinelli" in comm_lower:
            return "gas"
        elif "station u 9240101, 06plascassier" in comm_lower:
            return "gas"
        elif "intermarche dac, gattieres, fra, achat vpay du 09/10/2021 a 08:26" in comm_lower:
            return "gas"
        elif "tankstelle" in comm_lower:
            return "gas"

        elif "intermarche, cuers" in comm_lower:
            return "gift"

        elif "kiabi" in comm_lower:
            return "clothes"
        elif "chaussea" in comm_lower:
            return "clothes"
        elif "decathlon" in comm_lower:
            return "clothes"
        elif "jules" in comm_lower and "2980914" in comm_lower:
            return "clothes"

        elif "ville de luxembourg" in comm_lower:
            return "taxes and utilities"
        elif "edf" in comm_lower:
            return "taxes and utilities"
        elif "foyer assurances sa" in comm_lower:
            return "taxes and utilities"
        elif "tdo, 3 rue jean piret" in comm_lower:
            return "taxes and utilities"

        elif "sfr" in comm_lower:
            return "phone"
        elif "orange communications luxembourg" in comm_lower:
            return "phone"

        elif "billaudel" in comm_lower:
            return "rent"

        else:
            return np.nan

    def generate_debit_df(self):
        debit = self.data.copy()
        debit = debit[np.all([debit["Montant"] < 0, debit["Type de mouvement"]
                              != "DECOMPTE VISA"], axis=0)].copy()
        debit["spent"] = debit.Montant * -1
        debit.drop(columns="Montant", inplace=True, errors="ignore")
        debit = debit.reset_index(drop=True)
        debit["label"] = debit["Communication"].apply(
            self.make_initial_categories)

        self.debit = debit

    def plot_dump(self):
        pass
        # px.bar(exp.sort_values("spent", ascending=False), y="spent",
        #        color="spent", color_continuous_scale="Bluered")
        # px.scatter(debit, x="Date d'opération", y="spent")

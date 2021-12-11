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

        cat_dict = {}

        tolls_list = ["escota vinci",
                      "aprr"]

        food_fancy = ["maxicoffee sud",
                      "pizza",
                      "pizzeria",
                      "mirage sarl 4173455",  # hop store (bar)
                      "rpa sta acro3",  # beer with generationZ
                      "colgan, antibes"]  # The Duke bar

        food_basic = ["carrefourmarket, mougins",
                      "leader, antibes",
                      "sm casino cs898, biot",
                      "carrefour niceli",
                      "sc baghera, 06mougins",
                      "sm casino cs638",
                      "helios crou 1 sc2341529", ]

        tech_list = ["boulanger, mandelieu",
                     "darty"]

        cash_list = ["retrait bancomat"]

        gas_list = ["esso moulieres",
                    "q8 martinelli",
                    "station u 9240101, 06plascassier",
                    "intermarche dac, gattieres, fra, achat vpay du 09/10/2021 a 08:26",
                    "tankstelle"]

        gift_list = ["intermarche, cuers"]

        clothes_list = ["kiabi",
                        "chaussea",
                        "decathlon"]

        taxes_and_utilities_list = ["ville de luxembourg",
                                    "edf",
                                    "foyer assurances sa",
                                    "tdo, 3 rue jean piret"]

        phone_list = ["sfr",
                      "orange communications luxembourg"]

        rent_list = ["billaudel"]

        health_list = ["pharmacie"]

        # if "jules" in comm_lower and "2980914" in comm_lower:
        #     return "clothes"

        cat_dict["tolls"] = tolls_list
        cat_dict["food_non_groceries"] = food_fancy
        cat_dict["groceries"] = food_basic
        cat_dict["tech"] = tech_list
        cat_dict["cash"] = cash_list
        cat_dict["gas"] = gas_list
        cat_dict["gift"] = gift_list
        cat_dict["clothes"] = clothes_list
        cat_dict["taxes_and_utilities"] = taxes_and_utilities_list
        cat_dict["phone"] = phone_list
        cat_dict["rent"] = rent_list
        cat_dict["health"] = health_list


        for category, category_list in cat_dict.items():
            for marker in category_list:
                if marker in comm_lower:
                    print(f"{category} is returned for {marker}")
                    return category
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

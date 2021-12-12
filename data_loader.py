import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataLoader():
    def __init__(self, PATH="expenses.csv") -> None:
        self.PATH = PATH
        self.load_data()
        self.generate_debit_df()

    def load_data(self):
        "load initial data"
        self.initial_data = pd.read_csv(self.PATH, header=3, sep=";")
        data = self.initial_data.copy()
        new_col_dict = {"Date d'opération": "date",
                        "Type de mouvement": "transfer_type",
                        "Montant": "amount",
                        "Devise": "currency",
                        "Communication": "communication"}
        data = data.rename(columns=new_col_dict)
        data["amount"] = data["amount"].str.replace(",", ".")
        data["amount"] = data["amount"].astype(float)
        data = data.convert_dtypes()
        data["date"] = pd.to_datetime(data["date"])
        self.data = data

    def make_initial_categories(self, comm):
        "categorize expenses based on communication"
        try:
            comm_lower = comm.lower()
        except Exception as e:
            return "taxes_and_utilities"

        cat_dict = {}
        tolls_list = ["escota vinci",
                      "aprr"]
        food_fancy_list = ["maxicoffee sud",
                           "pizza",
                           "pizzeria",
                           "mirage sarl 4173455",  # hop store (bar)
                           "rpa sta acro3",  # beer with generationZ
                           "colgan, antibes",  # The Duke bar
                           "le blue whales",  # integration
                           "esteban cafe",
                           "les 3 diables",
                           "groovin, nice"]
        food_basic_list = ["carrefourmarket, mougins",
                           "leader, antibes",
                           "sm casino cs898, biot",
                           "carrefour niceli",
                           "sc baghera, 06mougins",
                           "sm casino cs638",
                           "helios crou 1 sc2341529",
                           "k e n, 06mougins",
                           "auchan cloche d'or"]
        tech_list = ["boulanger, mandelieu",
                     "darty",
                     "microsoft payments"]
        cash_list = ["retrait bancomat"]
        car_list = ["esso moulieres",
                    "q8 martinelli",
                    "station u 9240101, 06plascassier",
                    "intermarche dac, gattieres, fra, achat vpay du 09/10/2021 a 08:26",
                    "tankstelle",
                    "intermarche dac",
                    "total 4973475",
                    "total 4382876",
                    "aral station",
                    "frais voiture",
                    "bp capellen",
                    "q8 gasperich",
                    "snct livange",
                    "shell echternach",
                    "bsca parking"]
        gift_list = ["intermarche, cuers",
                     "floriane, 06mougins"]
        clothes_list = ["kiabi",
                        "chaussea",
                        "decathlon",
                        "jules 2980914"]
        taxes_and_utilities_list = ["ville de luxembourg",
                                    "edf",
                                    "foyer assurances sa",
                                    "tdo, 3 rue jean piret",
                                    "leo s.a."]
        phone_list = ["sfr",
                      "orange communications luxembourg"]
        rent_list = ["billaudel",
                     "rent + bill"]
        health_list = ["pharmacie",
                       "laboratoire 4336511",
                       "dr poyet",
                       "pcr test"]
        travel_list = ["ratp",
                       "sncf",
                       "maison bichon",
                       "sarl cvfm 2442422",
                       "groupe bigbang",
                       "louisblanc",
                       "sc-mamma",
                       "waiz naritake",
                       "le blue whales",
                       "ste claire borne2310551",
                       "aparcamiento baluarte",
                       "albergue casa baztan",
                       "est serv herno ",
                       "vieja fontana de oro",
                       "intermarche",
                       "intermarche laverie",
                       "cafe du passage 2933137",
                       "station alvea",
                       "prima fabbrica",
                       "le moderne",
                       "l esplanade",
                       "chez tonton",
                       "la cremaillere",
                       "sc horodateur 2331310",
                       "caves du pere augus",
                       "photomaton 2508911",
                       "photomaton 2508911",
                       "horodateurs 2384313",
                       "la creperie de jean",
                       "leclerc station",
                       "qpf rabine 2950082",
                       "citedia hoche",
                       "est serv herno",
                       "chez cloclo",
                       "vt mt st michel",
                       "stationnem horod2350392",
                       "cora saint malo"]

        tuition_fees_list = ["first year msc"]

        cat_dict["tolls"] = tolls_list
        cat_dict["food_fancy"] = food_fancy_list
        cat_dict["food_basic"] = food_basic_list
        cat_dict["tech"] = tech_list
        cat_dict["cash"] = cash_list
        cat_dict["car"] = car_list
        cat_dict["gift"] = gift_list
        cat_dict["clothes"] = clothes_list
        cat_dict["taxes_and_utilities"] = taxes_and_utilities_list
        cat_dict["phone"] = phone_list
        cat_dict["rent"] = rent_list
        cat_dict["health"] = health_list
        cat_dict["travel"] = travel_list
        cat_dict["tuition_fees"] = tuition_fees_list

        for category, category_list in cat_dict.items():
            for marker in category_list:
                if marker in comm_lower:
                    return category
        return np.nan

    def generate_debit_df(self):
        "Generate a dataframe with debits only"

        # Create dataframe from self.data
        debit = self.data.copy()
        debit = debit[np.all([debit["amount"] < 0,
                              debit["transfer_type"] != "DECOMPTE VISA"], axis=0)]
        debit = debit.reset_index(drop=True)
        debit["spent"] = debit.amount * -1

        # drop currency column if all debits are euros and drop amount column
        assert np.all(debit["currency"].unique() == ["EUR"]
                      ), "Some currencies are different from 'EUR'"

        debit.drop(columns=["amount", "currency"],
                   inplace=True,
                   errors="ignore")

        # Make labels
        debit["label"] = debit["communication"].apply(
            self.make_initial_categories)

        self.debit = debit

    def label_encode_debit(self):
        transfer_type_enc = LabelEncoder().fit_transform(self.debit["transfer_type"])
        label_enc = LabelEncoder().fit(self.debit["label"])
        self.debit["label"] = label_enc.transform(self.debit["label"])
        self.debit["transfer_type"] = transfer_type_enc


    def split_date(self):
        self.debit["year"] = pd.DatetimeIndex(self.debit["date"]).year
        self.debit["month"] = pd.DatetimeIndex(self.debit["date"]).month
        self.debit["day"] = pd.DatetimeIndex(self.debit["date"]).day
        self.debit["dayofweek"] = pd.DatetimeIndex(self.debit["date"]).dayofweek
        self.debit = self.debit.drop(columns=["date"])

    def plot_dump(self):
        pass
        # px.bar(exp.sort_values("spent", ascending=False), y="spent",
        #        color="spent", color_continuous_scale="Bluered")
        # px.scatter(debit, x="Date d'opération", y="spent")

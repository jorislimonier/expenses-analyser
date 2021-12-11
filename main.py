# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import plotly.express as px
import numpy as np
import pandas as pd

data = pd.read_csv("expenses.csv", header=3, sep=";")
data["Montant"] = data["Montant"].str.replace(",", ".")
data["Montant"] = data["Montant"].astype(float)
data = data.convert_dtypes()
data["Date d\'opération"] = pd.to_datetime(data["Date d\'opération"])


data.tail()


# %%
debit = data[np.all([data["Montant"] < 0, data["Type de mouvement"]
                    != "DECOMPTE VISA"], axis=0)].copy()
debit["spent"] = debit.Montant * -1
debit.drop(columns="Montant", inplace=True, errors="ignore")
debit = debit.reset_index(drop=True)
debit


# %%
for row in debit.iterrows():
    print(type(row))
# %%
exp = pd.DataFrame(
    columns=["spent"],
    # add categories here
    index=["groceries", "tolls", "crous_food", "food_non_groceries",
           "health", "tech", "clothes", "cash", "gas", "gift", "taxes and utilities", "phone", "rent", "other"]
)
exp["spent"] = 0
debit["label"] = np.nan


def categorize(comm):
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


debit["label"] = debit["Communication"].apply(categorize)
debit["label"].value_counts(dropna=False)

# %%
px.bar(exp.sort_values("spent", ascending=False), y="spent",
       color="spent", color_continuous_scale="Bluered")


# %%
# pd.to_datetime(data["Date d'opération"])
px.scatter(debit, x="Date d'opération", y="spent")


# %%

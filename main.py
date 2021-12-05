# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import pandas as pd

data = pd.read_csv("/home/joris/Downloads/Export_Mouvements_Cpte courant Green Code 18-30 Study.csv", header=3, sep=";")
data["Montant"] = data["Montant"].str.replace(",", ".")
data["Montant"] = data["Montant"].astype(float)
data = data.convert_dtypes()
data["Date d\'opération"] = pd.to_datetime(data["Date d\'opération"])


data.dtypes


# %%
debit = data[np.all([data["Montant"] < 0, data["Type de mouvement"] != "DECOMPTE VISA"], axis=0)].copy()
debit["spent"] = debit.Montant * -1
debit.drop(columns="Montant", inplace=True, errors="ignore")

debit["Type de mouvement"].unique()
debit.head()


# %%
exp = pd.DataFrame(
    columns=["spent"],
    # add categories here
    index=["groceries", "tolls", "crous_food", "food_non_groceries",
           "health", "tech", "clothes", "cash", "gas", "gift", "taxes and utilities", "phone", "rent", "other"]
)
exp["spent"] = 0

# categorize your expenses
for row in debit.iterrows():
    info = row[1]
    comm_lower = info["Communication"].lower()
    if "escota vinci" in comm_lower:
        exp.loc["tolls", "spent"] += info["spent"]
    elif "aprr" in comm_lower:
        exp.loc["tolls", "spent"] += info["spent"]

    elif "helios crou 1 sc2341529" in comm_lower:
        exp.loc["crous_food", "spent"] += info["spent"]

    elif "pharmacie" in comm_lower:
        exp.loc["health", "spent"] += info["spent"]

    elif "maxicoffee sud" in comm_lower:
        exp.loc["food_non_groceries", "spent"] += info["spent"]
    elif "pizza" in comm_lower:
        exp.loc["food_non_groceries", "spent"] += info["spent"]
    elif "pizzeria" in comm_lower:
        exp.loc["food_non_groceries", "spent"] += info["spent"]
    elif "mirage sarl 4173455" in comm_lower:  # hop store (bar)
        exp.loc["food_non_groceries", "spent"] += info["spent"]
    elif "rpa sta acro3" in comm_lower:  # beer with generationZ
        exp.loc["food_non_groceries", "spent"] += info["spent"]
    elif "colgan, antibes" in comm_lower:  # The Duke bar
        exp.loc["food_non_groceries", "spent"] += info["spent"]

    elif "carrefourmarket, mougins" in comm_lower:
        exp.loc["groceries", "spent"] += info["spent"]
    elif "leader, antibes" in comm_lower:
        exp.loc["groceries", "spent"] += info["spent"]
    elif "sm casino cs898, biot" in comm_lower:
        exp.loc["groceries", "spent"] += info["spent"]
    elif "carrefour niceli" in comm_lower:
        exp.loc["groceries", "spent"] += info["spent"]
    elif "sc baghera, 06mougins" in comm_lower:
        exp.loc["groceries", "spent"] += info["spent"]
    elif "sm casino cs638" in comm_lower:
        exp.loc["groceries", "spent"] += info["spent"]

    elif "boulanger, mandelieu" in comm_lower:
        exp.loc["tech", "spent"] += info["spent"]
    elif "darty" in comm_lower:
        exp.loc["tech", "spent"] += info["spent"]

    elif "retrait bancomat" in comm_lower:
        exp.loc["cash", "spent"] += info["spent"]

    elif "esso moulieres" in comm_lower:
        exp.loc["gas", "spent"] += info["spent"]
    elif "q8 martinelli" in comm_lower:
        exp.loc["gas", "spent"] += info["spent"]
    elif "station u 9240101, 06plascassier" in comm_lower:
        exp.loc["gas", "spent"] += info["spent"]
    elif "intermarche dac, gattieres, fra, achat vpay du 09/10/2021 a 08:26" in comm_lower:
        exp.loc["gas", "spent"] += info["spent"]
    elif "tankstelle" in comm_lower:
        exp.loc["gas", "spent"] += info["spent"]

    elif "intermarche, cuers" in comm_lower:
        exp.loc["gift", "spent"] += info["spent"]

    elif "kiabi" in comm_lower:
        exp.loc["clothes", "spent"] += info["spent"]
    elif "chaussea" in comm_lower:
        exp.loc["clothes", "spent"] += info["spent"]
    elif "decathlon" in comm_lower:
        exp.loc["clothes", "spent"] += info["spent"]
    elif "jules" in comm_lower and "2980914" in comm_lower:
        exp.loc["clothes", "spent"] += info["spent"]

    elif "ville de luxembourg" in comm_lower:
        exp.loc["taxes and utilities", "spent"] += info["spent"]
    elif "edf" in comm_lower:
        exp.loc["taxes and utilities", "spent"] += info["spent"]
    elif "foyer assurances sa" in comm_lower:
        exp.loc["taxes and utilities", "spent"] += info["spent"]
    elif "tdo, 3 rue jean piret" in comm_lower:
        exp.loc["taxes and utilities", "spent"] += info["spent"]

    elif "sfr" in comm_lower:
        exp.loc["phone", "spent"] += info["spent"]
    elif "orange communications luxembourg" in comm_lower:
        exp.loc["phone", "spent"] += info["spent"]

    elif "billaudel" in comm_lower:
        exp.loc["rent", "spent"] += info["spent"]

    else:
        print(f"""{info["Communication"]}\n{info["spent"]}\n""")
        exp.loc["other", "spent"] += info["spent"]

exp


# %%
import plotly.express as px
px.bar(exp.sort_values("spent", ascending=False), y="spent", color="spent", color_continuous_scale="Bluered")


# %%
# pd.to_datetime(data["Date d'opération"])
px.scatter(debit, x="Date d'opération", y="spent")


# %%




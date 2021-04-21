import pandas as pd
import numpy as np

data = pd.read_excel("/home/joris/Downloads/downloaded_file_21-04_04-57-35.xlsx", header=3)

def clean_colnames():
    col = data.columns[0]
    col = col.replace("Ã©", "é")
    return col.split(";")


def row_cleaning(row):
    row_nonan = list(row[np.logical_not(pd.isnull(row))])
    core = ".".join([row_nonan[0], row_nonan[1]])
    core = core.split(';')
    communication = [core.pop()] + row_nonan[2:]
    return core + [communication]

df = pd.DataFrame(index=data.index,columns=clean_colnames())

for i in range(len(data)):
    row = np.array(data.iloc[i])
    df.iloc[i] = row_cleaning(row)

df
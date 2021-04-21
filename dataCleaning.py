import pandas as pd
import numpy as np

class DataCleaning():
    def __init__(self):
        self.data = pd.read_excel("/home/joris/Downloads/downloaded_file_21-04_04-57-35.xlsx", header=3)

    def clean_colnames(self):
        col = self.data.columns[0]
        col = col.replace("Ã©", "é")
        return col.split(";")


    def row_cleaning(self, row):
        row_nonan = list(row[np.logical_not(pd.isnull(row))])
        core = ".".join([row_nonan[0], row_nonan[1]])
        core = core.split(';')
        communication = [core.pop()] + row_nonan[2:]
        return core + [communication]

    def create_clean_df(self):
        df = pd.DataFrame(index=self.data.index,columns=self.clean_colnames())
        for i in range(len(self.data)):
            row = np.array(self.data.iloc[i])
            df.iloc[i] = self.row_cleaning(row)
        return df
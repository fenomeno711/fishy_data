import pandas as pd


def load(path):
    raw_data = pd.read_csv(path, sep =",")
    print(len(raw_data),'Datensätze importiert.')
    return raw_data




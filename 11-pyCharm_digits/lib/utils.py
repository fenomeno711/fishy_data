import pandas as pd


def read_csv(path):
    raw_data = pd.read_csv(path)
    return raw_data



import pandas as pd


def load(path):
    raw_data = pd.read_csv(path)
    return raw_data



#imports
from sklearn.model_selection import train_test_split


def clean(raw_data):
    cleaned_data = raw_data.dropna(axis=1, thresh=2)
    print(len(cleaned_data),'Datensätze verbleiben nach Cleaning.')
    return cleaned_data


def split(cleaned_data,ratio):
    data_train, data_test = train_test_split(cleaned_data, test_size=ratio, random_state=11)
    print('Aufteilung: Train:', len(data_train), 'Test:', len(data_test))
    return data_train,data_test

def set_target(data_train, data_test, column_head):
    y_train = data_train[column_head]
    X_train = data_train.drop(columns=[column_head])
    y_test = data_test[column_head]
    X_test = data_test.drop(columns=[column_head])
    return X_train, y_train, X_test, y_test

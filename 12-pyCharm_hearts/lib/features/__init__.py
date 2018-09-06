#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def clean(raw_data):
    clean_data = raw_data.dropna(axis=0, how='any')
    clean_data = clean_data.reset_index(drop=True)
    print(len(clean_data),'Datensätze verbleiben nach Cleaning.')
    return clean_data


def split(cleaned_data,ratio):
    data_train, data_test = train_test_split(cleaned_data, test_size=ratio, random_state=11)
    # data_train[columns] = cleaned_data[columns]
    print('Aufteilung: Train:', len(data_train), 'Test:', len(data_test))
    return data_train,data_test


def set_target(data_train, data_test, column_head):
    y_train = data_train[column_head]
    X_train = data_train.drop(columns=[column_head])
    y_test = data_test[column_head]
    X_test = data_test.drop(columns=[column_head])
    return X_train, y_train, X_test, y_test


def encode_binary(df):
    df.loc[:,'AHD'] = 1.0 * (df.loc[:,'AHD']=='Yes')
    return df


def encode_category(df, column_head):
    values, labels = df.loc[:,column_head].factorize()
    enc = OneHotEncoder()
    new_df = pd.DataFrame((enc.fit_transform(values.reshape(-1,1))).toarray())
    new_df.columns = column_head+'_'+labels
    df = pd.concat((df, new_df), axis=1, sort=True)
    df = df.drop(columns=[column_head],axis=1)
    return df



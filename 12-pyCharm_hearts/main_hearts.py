import loader
import features
import sklearn.linear_model as lm
import classifier

import pandas as pd



import evaluation


if __name__ == '__main__':

    raw_data = loader.load("../data/heart/Heart.csv")

    clean_data = features.clean(raw_data)
    clean_data = features.encode_binary(clean_data)
    clean_data = features.encode_category(clean_data, 'ChestPain')
    clean_data = features.encode_category(clean_data, 'Thal')

    data_train,data_test = features.split(clean_data,0.2)

    X_train, y_train, X_test, y_test = features.set_target(data_train, data_test, 'AHD')

    logReg = lm.LogisticRegression()
    print(clean_data.head())
    classifier.fit(logReg, X_train, y_train)

    pred, pred_proba = classifier.predict(logReg, X_test)

    evaluation.print_errors(y_test, pred)

    #print(len(data_train), len(data_test))



    #print(clean_data)
    # cleaned_train_data = data_train.dropna(axis=1, thresh=2)
    #
    # input_data = cleaned_train_data.iloc[:, 1:].values
    # targets = cleaned_train_data[0].values
    #
    # input_data2 = features.zip_codes.multires(input_data)
    #
    # # log reg with simple feature set
    # print("Evaluating simple feature set")
    # log_reg = lm.SGDClassifier(n_jobs=1, loss="log", max_iter = 50)
    #
    # classifier.fit(log_reg, input_data, targets)
    # pred, pred_proba = classifier.predict(log_reg, input_data)
    #
    # evaluation.print_errors(targets, pred)
    # print("")
    #
    # # log reg with advanced feature set
    # print("Evaluating modified feature set")
    # log_reg2 = lm.SGDClassifier(n_jobs=1, loss="log", max_iter=50)
    #
    # classifier.fit(log_reg2, input_data2, targets)
    # pred, pred_proba = classifier.predict(log_reg2, input_data2)
    #
    # evaluation.print_errors(targets, pred)
    #mychange
    # ksdjfskjf
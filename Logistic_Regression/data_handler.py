import pandas as pd
import numpy as np
import random
def load_dataset():
    """
    function for reading data from csv
    and processing to return a 2D feature matrix and a vector of class
    :return:
    """
    data=pd.read_csv("data_banknote_authentication.csv")
    df=pd.DataFrame(data)
    x_data=df.iloc[:,:-1]
    #print(x_data)
    y_data=df.iloc[:,-1]
    #print(y_data)
    # todo: implement
    return x_data.to_numpy(), y_data.to_numpy()


def split_dataset(X, y, test_size, shuffle):
    if shuffle==True:
        arr=np.arange(len(y))
        np.random.shuffle(arr)
        X=X[arr]
        y=y[arr]
        # print("eta X")
        # print(X)
        # print("Eta y")
        # print(y)
    train_size=1-test_size
    train_size=int(train_size*len(y))
    """
    function for spliting dataset into train and test
    :param X:
    :param y:
    :param float test_size: the proportion of the dataset to include in the test split
    :param bool shuffle: whether to shuffle the data before splitting
    :return:
    """
    # todo: implement.

    X_train=X[:train_size]
    y_train=y[:train_size]
    X_test=X[train_size:]
    y_test=y[train_size:]
    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):

    """
    Randomly sample with replacement
    Size of sample will be same as input data
    :param X:
    :param y:
    :return:
    """
    # todo: implement
    # X_sample=np.zeros_like(X)
    # y_sample=np.zeros_like(y)
    # i=0
    # while(i<len(y)):
    #     index=random.randint(0,(len(y)-1))
    #     X_sample[i]=(X[index])
    #     y_sample[i]=(y[index])
    #     i=i+1
    range=len(y)
    indexes=np.arange(0,range).tolist()
    #random.shuffle(indexes)
    idx=np.random.choice(indexes,range)
    X_sample=X[idx]
    y_sample=y[idx]
    assert X_sample.shape == X.shape
    assert y_sample.shape == y.shape
    return X_sample, y_sample

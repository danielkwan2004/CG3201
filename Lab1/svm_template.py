import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

np.random.seed(0)

class SVM:
    def __init__(self, C=1.0, learning_rate=0.1, max_iter=1000, batch_size=64):
        self.w = None
        self.b = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

class MultiClassSVM:
    def __init__(self, C=1.0, learning_rate=0.1, max_iter=1000, batch_size=64):
        self.labels = None
        self.classifiers = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
        
def load_data(binary=False):
    df_white = pd.read_csv("winequality-white.csv")
    df_red = pd.read_csv("winequality-red.csv")
    data_white = np.array([row.split(';') for row in df_white.iloc[:, 0].values])
    data_red = np.array([row.split(';') for row in df_red.iloc[:, 0].values])

    data = np.concatenate((data_white, data_red))
    
    X = data[:, :-1].astype('double')
    y = data[:, -1].astype('int')

    if binary:
        y = np.where(y > 5, 1, -1)

    # Shuffle
    indices = np.random.permutation(X.shape[0])
    X = X[indices]
    y = y[indices]

    # Split training set and test set
    Xtrain = X[:4800]   # dimension: 4800 x 11
    Xtest = X[4800:]    # dimension: 1697 x 11
    ytrain = y[:4800]   # dimension: 4800 x 1
    ytest = y[4800:]    # dimension: 1697 x 1

    # Normalization
    mu, sigma = Xtrain.mean(axis=0), Xtrain.std(axis=0)
    Xtrain = (Xtrain - mu) / sigma
    Xtest = (Xtest - mu) / sigma

    # Print data info
    print("Number of training samples: %d, number of test samples: %d" %(len(ytrain), len(ytest)))
    unique, counts = np.unique(ytrain, return_counts=True)
    print("Labels: ", unique)
    print("Number of training samples in each class:")
    print(counts)
    print()

    return Xtrain, ytrain, Xtest, ytest

def plot(Xtrain, ytrain, Xtest, ytest):
    pass

def confusionMatrix(ytrue, ypred):
    pass

def __main__():
    
    # SVM for binary classification
    print("---- Binary classification ----")
    Xtrain, ytrain, Xtest, ytest = load_data(binary=True)
    
    print("Training accuracy: ")
    print("Test accuracy: ")

    # SVM for multi-class classification
    print()
    print("---- Multi-class classification ----")
    
    Xtrain, ytrain, Xtest, ytest = load_data(binary=False)

    print("Training accuracy: ")
    print("Test accuracy: ")

    print("Confusion matrix:")

if __name__ == '__main__':
    __main__()

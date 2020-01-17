import numpy as np
from sklearn.model_selection import train_test_split

class DataLoader():
    def __init__(self, features_dir=None, labels_dir=None):
        self.X, self.Y = None, None
        self.X_train, self.Y_train = None, None
        self.X_val, self.Y_val = None, None
        self.X_test, self.Y_test = None, None
        
        if features_dir and labels_dir:
            self.load(features_dir, labels_dir)

    def load(self, features_dir, labels_dir):
        self.X = np.load(features_dir)
        self.Y = np.load(labels_dir)

    def split_train_val_test(self, train=0.7, val=0.2, test=0.1):
        X, X_test, Y, Y_test = train_test_split(self.X, self.Y, test_size=test)
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=val/(1-test))
        self.X_train, self.Y_train = X_train, Y_train
        self.X_val, self.Y_val = X_val, Y_val
        self.X_test, self.Y_test = X_test, Y_test

    def preprocess(self, rescale=False, center=False, standardize=True):
        if rescale:
            self.X_train = self.X_train / 255.0
            self.Y_train = self.Y_train / 255.0
            self.X_val = self.X_val / 255.0
            self.Y_val = self.Y_val / 255.0
            self.X_test = self.X_test / 255.0
            self.Y_test = self.Y_test / 255.0
       
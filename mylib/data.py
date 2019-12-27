import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class Data:
    # Load Adult dataset and seperate to features(X) and target(y)
    def __init__(self, path='data/adult.csv'):
        df = shuffle(pd.read_csv(path))
        df = self.clean(df)

        self.y = df.pop('income')
        self.X = df

        # Label encode y
        self.y_encoder = LabelEncoder()
        self.y = self.y_encoder.fit_transform(self.y)

        # One Hot encode X
        self.X = pd.get_dummies(self.X)

        for name in self.X.columns:
            if self.X[name].dtype == 'object':
                self.X[name] = self.X[name].astype('category')

    def clean(self, df):
        return df.replace('?', np.nan).dropna().drop('fnlwgt', axis=1)

    def train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2)
        y_train = pd.Series(y_train, index=X_train.index)
        y_test = pd.Series(y_test, index=X_test.index)
        return (X_train, X_test, y_train, y_test)

import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import LabelEncoder


# Preprocessing Adult dataset
class Data:
    # Load Adult dataset and seperate to features(X) and target(y)
    def __init__(self, path='data/adult.csv'):
        df = pd.read_csv(path)
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1:]
    
    # Apply Label Encoding on features(X) and target(y)
    # and Convert DataFrame to Numpy type
    def clean(self):
        X = self.label_encoding(self.X)
        y = self.label_encoding(self.y)
        return (X.values, y.values)

    # Apply Label Encoding on non-numeric columns
    def label_encoding(self, df):
        labelencoder = LabelEncoder()
        for col in df.columns:
            if(not is_numeric_dtype(df[col])):
                df[col] = labelencoder.fit_transform(df[col])
        return df
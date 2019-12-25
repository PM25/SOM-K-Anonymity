import pandas as pd
import np
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Preprocessing Adult dataset


class Data:
    # Load Adult dataset and seperate to features(X) and target(y)
    def __init__(self, path='data/adult.csv'):
        df = shuffle(pd.read_csv(path))
        df = self.clean(df)

        self.y = df.pop('income')
        self.X = df

    def clean(self, df):
        return df.replace(' ?', np.nan).dropna()

    def train_test_split(self):
        pass

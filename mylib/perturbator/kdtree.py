import pandas as pd
from minisom import MiniSom
from .perturbator import Perturbator

class Mondrian(Perturbator):
    def __init__(self, quasi_identifiers):
        self.quasi_identifiers = quasi_identifiers

    def is_categorical(self, df, column):
        return str(df[column].dtype) == 'category'

    def partite(self, df, partition, column):
        """partite the df into two partitions.

            returns: A tuple of df index.
        """
        df_partition = df[column][partition]
        if self.is_categorical(df, column):
            values = list(df_partition.unique())
            left = df_partition.isin(values[:len(values) // 2])
            right = df_partition.isin(values[len(values) // 2:])
            return (df_partition[left].index, df_partition[right].index)
        else:
            median = df_partition.median()
            return (df_partition[df_partition < median].index, df_partition[df_partition >= median].index)

    def get_spans(self, df, partition):
        """get each column's span
        """
        span = {}
        for column in self.quasi_identifiers:
            df_partition = df[column][partition]
            if self.is_categorical(df, column):
                span[column] = len(df_partition.unique())
            else:
                span[column] = df_partition.max() - df_partition.min()
        return sorted(span.items(), key=lambda x: x[1], reverse=True)

    def validate(self, df):
        pass

    def split(self, df):
        wip_partitions = [df.index]
        finished_partitions = []

        while len(wip_partitions) > 0:
            partition = wip_partitions.pop(0)
            for column, _ in self.get_spans(df, partition):
                lp, rp = self.partite(df, partition, column)

                # If either left part or right part cannot satisfied the K-anonymous condition
                # cancel the partion and try next column.
                if not self.validate(df.loc[lp]) or not self.validate(df.loc[rp]):
                    continue

                # If the partition is valid, continue to try next partition.
                wip_partitions.append(lp)
                wip_partitions.append(rp)
                break
            else:
                # If the partition cannot be partited anymore, put it into finished_partitions array.
                finished_partitions.append(partition)
        return finished_partitions

    def build_dataset(self, df, partitions):
        dfs = []
        for partition in partitions:
            dfp = df.loc[partition]
            for column in self.quasi_identifiers:
                if dfp[column].dtype == 'int64':
                    dfp[column] = dfp[column].mean()
                if str(dfp[column].dtype) == 'category':
                    dfp[column] = ','.join(list(dfp[column].unique()))
            dfs.append(dfp)
        return pd.concat(dfs)

    def perturbate(self, df):
        partitions = self.split(df)
        return self.build_dataset(df, partitions)


class K_Anonymity(Mondrian):
    def __init__(self, quasi_identifiers, k):
        self.k_anonymity = k
        super().__init__(quasi_identifiers)

    def is_k_anonymous(self, partition):
        return not (partition.shape[0] < self.k_anonymity)

    def validate(self, df):
        return self.is_k_anonymous(df)

    def __str__(self):
        return 'K Anonymity - {}'.format(self.k_anonymity)


class SOM_K_Anonymity(K_Anonymity):
    def __init__(self, quasi_identifiers, k, som_size=(150, 150)):
        self.som_columns = quasi_identifiers
        self.som_size = som_size
        super().__init__(['x axis', 'y axis'], k)

    def perturbate(self, df):
        df_som = df[self.som_columns]
        som = MiniSom(self.som_size[0], self.som_size[1], df_som.shape[1])
        som.train_random(df_som.values, 10)
        coordinates = [som.winner(series.to_numpy())
                       for index, series in df_som.iterrows()]
        df_coordinates = pd.DataFrame(
            coordinates, index=df.index, columns=['x axis', 'y axis'])
        df = pd.concat([df, df_coordinates], axis=1)
        df = super().perturbate(df)
        df.drop(['x axis', 'y axis'], axis=1, inplace=True)
        return df

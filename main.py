import pandas as pd
from mylib.helper import timeit, profile
from mylib.data import Data
from mylib.perturbator import K_Anonymity, SOM_K_Anonymity
from mylib.runner import TrainRunner

# Start from here!
if __name__ == "__main__":
    data = Data()
    X_train_origin, X_test_origin, y_train_origin, y_test_origin = data.train_test_split()

    runners = []

    # Creat a base model to compare the result.
    original_runner = TrainRunner('Original')
    runners.append(original_runner)

    quasi_identifiers = ['age', 'educational-num',
                         'capital-gain', 'capital-loss', 'hours-per-week']

    # Choose different k size.
    sizes = [5, 10, 15, 20, 30, 50]

    for size in sizes:
        runner = TrainRunner('K Anonymous(k={})'.format(
            size), [K_Anonymity(quasi_identifiers, size)])
        som_runner = TrainRunner('SOM KDTree(k={})'.format(size),
                                 [SOM_K_Anonymity(quasi_identifiers, size)])

        runners.append(runner)
        runners.append(som_runner)

        pf = profile(runners, X_train_origin, y_train_origin,
                     X_test_origin, y_test_origin)

        print(pd.DataFrame.from_dict(pf))

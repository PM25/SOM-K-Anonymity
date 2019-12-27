from .model import TrainingModel


class TrainRunner:
    """The class automate the preprocess data and train a model's process.
    
        Args:
            name: The name of the runner.
            perturbators: A list of perturbators, the perturbator should implements a perturbate
            function. The perturbator will be used in order when preprocess is called.
    """

    def __init__(self, name, perturbators=[]):
        self.perturbators = perturbators
        self.model = None
        self.name = name

    def __str__(self):
        return self.name

    def preprocess(self, X, y):
        for perturbator in self.perturbators:
            X = perturbator.perturbate(X)
        X, y = X.align(y, join='inner', axis=0)
        return (X, y)

    def fit(self, X, y):
        shape = (X.shape[1], )
        self.model = TrainingModel(shape)
        self.model.fit(X, y)

    def evaluate(self, X, y, print_report=False):
        if self.model is not None:
            report = self.model.evaluate(X, y, print_report=print_report)
            return report
        else:
            raise Exception('Must call fit before evaluate a model.')
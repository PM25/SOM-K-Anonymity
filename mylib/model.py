import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

class TrainingModel:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_shape=input_shape))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

    def fit(self, data, label):
        self.model.fit(data, label, epochs=1, batch_size=128, verbose=0)

    def predict(self, data):
        return self.model.predict_classes(data)
    
    def evaluate(self, X_test, y_test, print_report=True):
        y_predicted = self.predict(X_test)
        y_predicted_probs = self.model.predict_proba(X_test)
        if print_report:
            self.print_report(y_test, y_predicted, y_predicted_probs)
        else:
            accuracy = accuracy_score(y_test, y_predicted)
            report = classification_report(y_test, y_predicted, output_dict=True)
            auc_score = roc_auc_score(y_test, y_predicted_probs)
            matrix = confusion_matrix(y_test, y_predicted)

            return {
                'accuracy': accuracy,
                'auc_score': auc_score,
                **report['weighted avg'],
            }

    def print_report(self, test, predicted, predicted_probs):
        accuracy = accuracy_score(test, predicted)
        report = classification_report(test, predicted)
        matrix = confusion_matrix(test, predicted)

        print('Accuracy score: {:.5f}'.format(accuracy))
        print('-' * 20)
        print('Confusion Matrix:')
        print(matrix)
        print('-' * 20)
        print(report)
        print('-' * 20)
        print('AUC score: {:.5f}'.format(roc_auc_score(test, predicted_probs)))
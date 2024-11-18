import pandas as pd
from jsonschema.exceptions import best_match
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Function to encode data to TF-IDF vectors
def Data_encoder(train_data, valid_data, test_data):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_data['text'])
    X_valid = vectorizer.transform(valid_data['text'])
    X_test = vectorizer.transform(test_data['text'])

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(train_data['label'])
    y_valid = label_encoder.fit_transform(valid_data['label'])
    y_test = label_encoder.fit_transform(test_data['label'])
    return X_train, y_train, X_valid, y_valid, X_test, y_test


class Senti_SVM_model():
    def __init__(self, model = None, C_values = np.arange(1.7, 1.9, 0.01), kernel_values = ['linear', 'rbf']):
        '''
        Paras:
        - C-values: possible regulation coefficient of SVM
        - kernel_values: kernel of SVM
        '''
        # C_values = np.arange(0.5, 100, 0.5)
        # C_values = np.arange(1.5, 2.5, 0.1)
        self.C_values = C_values
        self.kernel_values = kernel_values

    def fit(self, X_train, y_train, X_valid, y_valid):
        best_model = None
        best_accuracy = 0

        for C in self.C_values:
            for kernel in self.kernel_values:
                model = SVC(C=C, kernel=kernel, random_state=42)
                model.fit(X_train, y_train)

                # test on validation set
                y_val_pred = model.predict(X_valid)
                accuracy = accuracy_score(y_valid, y_val_pred)
                print(f"C={C}, kernel={kernel}, Validation Accuracy={accuracy:.2f}")

                # keep the best
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
        print(f"Best Model: C={best_model.C}, kernel={best_model.kernel}, Validation Accuracy={best_accuracy:.2f}")
        self.model = best_model

    def evaluate(self, X_test, y_test):
        # evaluate
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.2f}")


if __name__ == '__main__':
    # load data
    train_data = pd.read_csv('nusax-main/datasets/sentiment/indonesian/train.csv')
    valid_data = pd.read_csv('nusax-main/datasets/sentiment/indonesian/valid.csv')
    test_data = pd.read_csv('nusax-main/datasets/sentiment/indonesian/test.csv')

    X_train, y_train, X_valid, y_valid, X_test, y_test = Data_encoder(train_data, valid_data, test_data)

    model = Senti_SVM_model()
    model.fit(X_train, y_train, X_valid, y_valid)
    model.evaluate(X_test, y_test)


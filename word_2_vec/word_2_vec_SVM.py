import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from gensim.models import KeyedVectors
import numpy as np


# Function to encode data to Word2Vec average embeddings
def Data_encoder_word2vec(data, word2vec, label_encoder=None, is_train=True):
    """
    Encodes text data into average Word2Vec embeddings and labels.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing 'text' and 'label' columns.
    - word2vec (KeyedVectors): Pre-trained Word2Vec model.
    - label_encoder (LabelEncoder or None): Encoder for labels (if available).
    - is_train (bool): Whether to fit the label encoder (used in training).
    
    Returns:
    - X (np.ndarray): Encoded text embeddings.
    - y (np.ndarray): Encoded labels.
    - label_encoder (LabelEncoder): Fitted label encoder (if training).
    """
    embedding_dim = word2vec.vector_size
    texts = data['text'].tolist()
    labels = data['label'].tolist()

    # Convert text to Word2Vec embeddings
    def text_to_embedding(text):
        tokens = text.split()
        embeddings = [word2vec[word] if word in word2vec else np.zeros(embedding_dim) for word in tokens]
        if len(embeddings) == 0:
            return np.zeros(embedding_dim)
        return np.mean(embeddings, axis=0)

    X = np.array([text_to_embedding(text) for text in texts])

    # Encode labels
    if label_encoder is None:
        label_encoder = LabelEncoder()
    if is_train:
        y = label_encoder.fit_transform(labels)
    else:
        y = label_encoder.transform(labels)
    
    return X, y, label_encoder


class Senti_SVM_model:
    """
    Sentiment analysis using SVM with Word2Vec embeddings.
    """
    def __init__(self, C_values=np.arange(1.0, 3.0, 0.2), kernel_values=['linear', 'rbf']):
        """
        Initialize the Senti_SVM_model with hyperparameters.

        Parameters:
        - C_values (list): Range of C values to try.
        - kernel_values (list): List of kernel types to try.
        """
        self.C_values = C_values
        self.kernel_values = kernel_values

    def fit(self, X_train, y_train, X_valid, y_valid):
        """
        Train the SVM model and select the best hyperparameters.

        Parameters:
        - X_train (np.ndarray): Training embeddings.
        - y_train (np.ndarray): Training labels.
        - X_valid (np.ndarray): Validation embeddings.
        - y_valid (np.ndarray): Validation labels.
        """
        best_model = None
        best_accuracy = 0

        for C in self.C_values:
            for kernel in self.kernel_values:
                model = SVC(C=C, kernel=kernel, random_state=42)
                model.fit(X_train, y_train)

                # Evaluate on validation data
                y_val_pred = model.predict(X_valid)
                accuracy = accuracy_score(y_valid, y_val_pred)
                print(f"C={C}, kernel={kernel}, Validation Accuracy={accuracy:.2f}")

                # Keep the best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model

        print(f"Best Model: C={best_model.C}, kernel={best_model.kernel}, Validation Accuracy={best_accuracy:.2f}")
        self.model = best_model

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.

        Parameters:
        - X_test (np.ndarray): Test embeddings.
        - y_test (np.ndarray): Test labels.
        """
        y_test_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_test_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_test_pred))


if __name__ == '__main__':
    # Paths to datasets and Word2Vec file
    word2vec_path = './cc.id.300.vec'
    train_file = '../nusax-main/datasets/sentiment/indonesian/train.csv'
    valid_file = '../nusax-main/datasets/sentiment/indonesian/valid.csv'
    test_file = '../nusax-main/datasets/sentiment/indonesian/test.csv'

    # Load Word2Vec embeddings
    print(f"Loading Word2Vec model from {word2vec_path}...")
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    print("Word2Vec model loaded!")

    # Load datasets
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)

    # Encode datasets
    X_train, y_train, label_encoder = Data_encoder_word2vec(train_data, word2vec)
    X_valid, y_valid, _ = Data_encoder_word2vec(valid_data, word2vec, label_encoder, is_train=False)
    X_test, y_test, _ = Data_encoder_word2vec(test_data, word2vec, label_encoder, is_train=False)

    # Train and evaluate the SVM model
    model = Senti_SVM_model()
    model.fit(X_train, y_train, X_valid, y_valid)
    model.evaluate(X_test, y_test)

import argparse
from torch.optim import AdamW
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from model_senti.Senti_DataLoader import built_curpus, SentiDataset
from model_senti.CNN_senti import CNNnet # Assuming CNN_senti is in cnn_senti_model.py
from model_senti.MLP_senti import MLPModel  # Assuming MLPModel is in mlp_senti_model.py
from model_senti.SVM_senti import *

# Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--max_length', type=int, default=1000, help="Maximum sequence length")
    parser.add_argument('--embedding_dim', type=int, default=512, help="Embedding dimension")
    parser.add_argument('--num_classes', type=int, default=3, help="Number of output classes")
    parser.add_argument('--model', type=str, default="MLP", choices=["MLP", "CNN","SVM"], help="Model type")
    return parser.parse_args()


# Training Function
def train():
    args = parse_args()

    # Load datasets
    train_file_path = 'nusax-main/datasets/sentiment/indonesian/train.csv'
    val_file_path = 'nusax-main/datasets/sentiment/indonesian/valid.csv'
    train_data = pd.read_csv(train_file_path, usecols=lambda col: col != 'Unnamed: 0')
    val_data = pd.read_csv(val_file_path, usecols=lambda col: col != 'Unnamed: 0')
    senti_word_2_index = built_curpus(train_data['text'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create DataLoaders
    train_dataset = SentiDataset(train_data, senti_word_2_index, args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = SentiDataset(val_data, senti_word_2_index, args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    if args.model == "MLP":
        model = MLPModel(len(senti_word_2_index), args.embedding_dim, args.num_classes, args.dropout)
    elif args.model == "CNN":
        model = CNNnet(len(senti_word_2_index), args.embedding_dim, args.num_classes, args.dropout)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (labels == outputs.argmax(dim=1)).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch + 1}, Train Loss: {total_loss / len(train_loader):.4f}, Train Acc: {correct / total:.4f}")

        # Validation
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                outputs = model(tokens)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()
                correct += (labels == outputs.argmax(dim=1)).sum().item()
                total += labels.size(0)

        print(f"Epoch {epoch + 1}, Val Loss: {total_loss / len(val_loader):.4f}, Val Acc: {correct / total:.4f}")
    return model
def test(model):
    test_file_path = 'nusax-main/datasets/sentiment/indonesian/test.csv'
    test_data = pd.read_csv(test_file_path, usecols=lambda col: col != 'Unnamed: 0')
    senti_word_2_index = built_curpus(test_data['text'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_dataset = SentiDataset(test_data, senti_word_2_index, max_length=1000)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            outputs = model(tokens)
            loss = loss_fn(outputs, labels)

            total_loss += loss.item()
            correct += (labels == outputs.argmax(dim=1)).sum().item()
            total += labels.size(0)

    print(f"Test Loss: {total_loss / len(test_loader):.4f}, Test Accuracy: {correct / total:.4f}")
    print(f"Model Structure: {model}")



def train_svm(train_file, valid_file, test_file):
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)

    X_train, y_train, X_valid, y_valid, X_test, y_test = Data_encoder(train_data, valid_data, test_data)
    model = Senti_SVM_model()
    model.fit(X_train, y_train, X_valid, y_valid)
    model.evaluate(X_test, y_test)

if __name__ == '__main__':
    args = parse_args()
    if args.model == 'SVM':
        train_svm('nusax-main/datasets/sentiment/indonesian/train.csv',
                  'nusax-main/datasets/sentiment/indonesian/valid.csv',
                  'nusax-main/datasets/sentiment/indonesian/test.csv')
    else:
        trained_model = train()  # For MLP or CNN
        test(trained_model)
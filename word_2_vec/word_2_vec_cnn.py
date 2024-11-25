import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gensim.models import KeyedVectors
import pandas as pd
import torch.nn.functional as F


class CNNnet(nn.Module):
    def __init__(self, embedding_dim, num_classes, dropout=0.2, kernel_num=100, kernel_sizes=[3, 4, 5]):
        super(CNNnet, self).__init__()
        self.conv1d_list = nn.ModuleList([
            nn.Conv2d(1, kernel_num, (kernel_size, embedding_dim))
            for kernel_size in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, num_classes)

    def conv_and_pool(self, x, conv):
        x = conv(x)  # Convolution
        x = F.relu(x.squeeze(3))  # Activation
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # Max pooling
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension: (batch_size, 1, max_length, embedding_dim)
        conv_outs = [self.conv_and_pool(x, conv) for conv in self.conv1d_list]
        x = torch.cat(conv_outs, dim=1)  # Concatenate outputs from all convolution layers
        x = self.dropout(x)  # Apply dropout
        x = self.fc1(x)  # Fully connected layer
        return x


class SentimentDataset(Dataset):
    def __init__(self, data, word2vec, max_length=100):
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()
        self.word2vec = word2vec
        self.max_length = max_length
        self.embedding_dim = word2vec.vector_size

    def __len__(self):
        return len(self.texts)

    def text_to_embedding(self, text):
        tokens = text.split()
        embeddings = [
            torch.tensor(self.word2vec[word], dtype=torch.float32) if word in self.word2vec else torch.zeros(self.embedding_dim)
            for word in tokens[:self.max_length]
        ]
        if len(embeddings) < self.max_length:
            embeddings += [torch.zeros(self.embedding_dim)] * (self.max_length - len(embeddings))
        return torch.stack(embeddings)

    def __getitem__(self, idx):
        text = self.text_to_embedding(self.texts[idx])
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        label = label_map[self.labels[idx]]
        return text, label


def train_cnn_for_sentiment(word2vec_path, train_file, val_file, test_file, max_length=100, batch_size=32, num_epochs=10, learning_rate=1e-3):
    print(f"Loading Word2Vec model from {word2vec_path}...")
    word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    print("Word2Vec model loaded!")

    train_data = pd.read_csv(train_file)
    val_data = pd.read_csv(val_file)
    test_data = pd.read_csv(test_file)

    train_dataset = SentimentDataset(train_data, word2vec, max_length)
    val_dataset = SentimentDataset(val_data, word2vec, max_length)
    test_dataset = SentimentDataset(test_data, word2vec, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    cnn_model = CNNnet(
        embedding_dim=word2vec.vector_size,
        num_classes=3,
        dropout=0.2
    ).to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        cnn_model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for texts, labels in train_loader:
            texts, labels = texts.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()
            outputs = cnn_model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss / len(train_loader):.4f}, Train Acc: {train_correct / train_total:.4f}")

        cnn_model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to('cuda'), labels.to('cuda')
                outputs = cnn_model(texts)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        print(f"Epoch {epoch + 1}/{num_epochs} - Val Loss: {val_loss / len(val_loader):.4f}, Val Acc: {val_correct / val_total:.4f}")

    # Test loop
    test_loss, test_correct, test_total = 0.0, 0, 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to('cuda'), labels.to('cuda')
            outputs = cnn_model(texts)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
            test_total += labels.size(0)

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Test Acc: {test_correct / test_total:.4f}")


if __name__ == "__main__":
    word2vec_path = "./cc.id.300.vec"
    train_file = "../nusax-main/datasets/sentiment/indonesian/train.csv"
    val_file = "../nusax-main/datasets/sentiment/indonesian/valid.csv"
    test_file = "../nusax-main/datasets/sentiment/indonesian/test.csv"
    train_cnn_for_sentiment(word2vec_path, train_file, val_file, test_file)


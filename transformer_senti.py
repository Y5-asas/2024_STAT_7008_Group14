import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from Senti_DataLoader import built_curpus
import os
from Visualization import get_confusion_matrix, plot_loss
from pathlib import Path



class SentiDataset(Dataset):
    def __init__(self, data, word_2_index, max_length=1000):
        self.texts = data['text'].tolist()
        self.labels = data['label'].tolist()
        self.word_2_index = word_2_index
        self.max_length = max_length
        self.target_map = {'positive': 1, 'negative': 0, 'neutral': 2}

    def tokenize_and_pad(self, text, word_2_index):
        tokenized = [word_2_index.get(word, word_2_index["<UNK>"]) for word in text.split()]
        tokenized = tokenized + [word_2_index["<EOS>"]]
        if len(tokenized) < self.max_length:
            tokenized += [word_2_index["<PAD>"]] * (self.max_length - len(tokenized))
        else:
            tokenized = tokenized[:self.max_length]
        return torch.tensor(tokenized, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.target_map[self.labels[idx]]

        text_tokens = self.tokenize_and_pad(text, self.word_2_index)
        return text_tokens, torch.tensor(label, dtype=torch.long)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerSentimentClassifier(nn.Module):
    def __init__(self, src_vocab_size, num_labels, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerSentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer, num_encoder_layers)
        self.fc = nn.Linear(d_model, num_labels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        src = self.pos_encoder(src)
        if src_mask is not None:
            src_mask = src_mask.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_mask)
        output = self.dropout(output.mean(dim=1))
        output = self.fc(output)
        return output


# Training and evaluation functions
def train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, ckpt_path):
    train_losses = []
    valid_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for tokens, labels in train_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            src_mask = (tokens == senti_word_2_index["<PAD>"])
            optimizer.zero_grad()
            outputs = model(tokens, src_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}")

        model.eval()
        total_loss = 0
        correct_predictions = 0
        best_accuracy = 0
        with torch.no_grad():
            for tokens, labels in valid_loader:
                tokens, labels = tokens.to(device), labels.to(device)
                src_mask = (tokens == senti_word_2_index["<PAD>"])
                outputs = model(tokens, src_mask)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=-1)
                correct_predictions += torch.sum(preds == labels)
            avg_valid_loss = total_loss / len(valid_loader)
            valid_losses.append(avg_valid_loss)
            accuracy = correct_predictions / len(valid_loader.dataset)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save(model.state_dict(), ckpt_path)
            print(f"Validation Loss: {avg_valid_loss}, Validation Accuracy: {accuracy}")
    return train_losses, valid_losses
# Test the model
def test(model, test_loader):
    model.eval()
    correct_predictions = 0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for tokens, labels in test_loader:
            tokens, labels = tokens.to(device), labels.to(device)
            src_mask = (tokens == senti_word_2_index["<PAD>"])
            outputs = model(tokens, src_mask)
            predictions = torch.argmax(outputs, dim=-1)
            correct_predictions += (predictions == labels).sum().item()
            all_labels.append(labels)
            all_preds.append(predictions)

    accuracy = correct_predictions / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy}")
    return all_labels, all_preds

if __name__ == '__main__':
    # Load data
    train_path = './nusax-main/datasets/sentiment/indonesian/train.csv'
    valid_path = './nusax-main/datasets/sentiment/indonesian/valid.csv'
    test_path = './nusax-main/datasets/sentiment/indonesian/test.csv'

    train_data = pd.read_csv(train_path)
    valid_data = pd.read_csv(valid_path)
    test_data = pd.read_csv(test_path)

    # Build vocabularies
    senti_word_2_index = built_curpus(train_data['text'])

    # Create datasets and dataloaders
    max_length = 200
    batch_size = 16

    train_dataset = SentiDataset(train_data, senti_word_2_index, max_length)
    valid_dataset = SentiDataset(valid_data, senti_word_2_index, max_length)
    test_dataset = SentiDataset(test_data, senti_word_2_index, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_model = TransformerSentimentClassifier(src_vocab_size=len(senti_word_2_index), num_labels=3, nhead = 8, num_encoder_layers=6, dropout=0.1).to(device)
    
    # Initialize model, criterion, and optimizer
    model = init_model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    # Train the model
    num_epochs = 100
    save_dir = './senti_ckpt21'
    os.makedirs(save_dir, exist_ok=True)
    ckpt_path = os.path.join(save_dir, 'best_model2.pth')
    train_losses, valid_losses = train(model, train_loader, valid_loader, criterion, optimizer, num_epochs, ckpt_path)

    # Test the model
    model = init_model
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    all_labels, all_preds = test(model, test_loader)
    map = {0: 'negative', 1: 'positive', 2: 'neutral'}
    all_labels = [map[label.item()] for labels in all_labels for label in labels]
    all_preds = [map[pred.item()] for preds in all_preds for pred in preds]
    get_confusion_matrix(all_labels, all_preds, save_path=Path(save_dir))
    plot_loss(train_losses, valid_losses,save_path=Path(save_dir))

import argparse
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from Senti_DataLoader import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training MLPModel")
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dropout', type=int, default=0.2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--max_length', type=int, default=1000)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--cpu', action='store_true', help='cpu training')

    args = parser.parse_args()

    return args

class MLP(nn.Module):
    '''
    naive fc layer
    '''

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.BatchNorm1d(out_dim),
                                 nn.Dropout(dropout),
                                 nn.ReLU(),
                                 )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, input_size, embedding_dim, num_classes, dropout, hidden_layers = [1024, 512, 128, 10]):
        super(MLPModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.fc0 = MLP(embedding_dim * 1000, hidden_layers[0], dropout)
        self.fcs = nn.ModuleList([MLP(hidden_layers[i - 1], hidden_layers[i], dropout) for i in range(1, len(hidden_layers))])
        self.fc1 = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        for fc in self.fcs:
            x = fc(x)
        x = self.fc1(x)
        return x

def train():
    train_file_path = 'nusax-main/datasets/sentiment/indonesian/train.csv'
    val_file_path = 'nusax-main/datasets/sentiment/indonesian/valid.csv'

    train_data = pd.read_csv(train_file_path, usecols=lambda col: col != 'Unnamed: 0')
    val_data = pd.read_csv(val_file_path, usecols=lambda col: col != 'Unnamed: 0')
    # Build vocabularies
    senti_word_2_index = built_curpus(train_data['text'])

    # Create Dataset and DataLoader
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SentiDataset(train_data, senti_word_2_index, args.max_length)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset = SentiDataset(val_data, senti_word_2_index, args.max_length)
    val_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = MLPModel(len(senti_word_2_index), args.embedding_dim, args.num_classes,args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    # scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs * len(train_dataloader), pct_start=0.25,
    #                        anneal_strategy='cos')
    loss_fn = nn.CrossEntropyLoss(reduction='mean')

    for epoch in range(args.epochs):
        model.train()
        # total loss of each epoch
        epoch_loss = 0.0
        acc = 0
        num = 0

        for tokens, labels in train_dataloader:
            tokens = tokens.to(device)
            labels = labels.to(device)
            pred = model(tokens)
            loss = loss_fn(pred, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # scheduler.step()
            acc += (labels == torch.argmax(pred, dim=1)).sum().item()
            num += len(labels)
            ls = loss.item()
            epoch_loss += ls
        avg_train_loss = epoch_loss / len(train_dataloader)

        print(f"Epoch {epoch + 1}/{args.epochs}, Training Loss: {avg_train_loss}")
        print(f"Epoch {epoch + 1}/{args.epochs}, Training Accuracy: {acc / num}")
        model.eval()
        epoch_loss = 0.0
        acc = 0
        num = 0
        with torch.no_grad():
            for tokens, labels in val_dataloader:
                tokens = tokens.to(device)
                labels = labels.to(device)
                pred = model(tokens)
                loss = loss_fn(pred, labels)

                acc += (labels == torch.argmax(pred, dim=1)).sum().item()
                num += len(labels)
                ls = loss.item()
                epoch_loss += ls

            print(f"Epoch {epoch + 1}/{args.epochs}, Validation Accuracy: {acc / num}")
            avg_val_loss = epoch_loss / len(val_dataloader)
            print(f"Epoch {epoch + 1}/{args.epochs}, Validation Loss: {avg_val_loss}")
            print('--------------------------------------------------')


if __name__ == '__main__':
    train()


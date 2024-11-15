import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from Data_loader import TranslationDataset, built_curpus
import pandas as pd
from cnn_model import CNNModel  # Replace or extend with different models as needed
from lstm_model import *
from CNN_1 import *
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train Machine Translation Model")
    parser.add_argument('--train_file', type=str, default='./nusax-main/datasets/mt/train.csv', help='Path to the training data file')
    parser.add_argument('--val_file', type=str, default='./nusax-main/datasets/mt/valid.csv', help='Path to the validation data file')
    parser.add_argument('--test_file', type=str, default='./nusax-main/datasets/mt/test.csv', help='Path to the test data file')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of embeddings')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate for optimizer')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length for input data')
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN','LSTM','CNN_1'], help='Type of model to use')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Base directory for logs and checkpoints')
    return parser.parse_args()

def setup_experiment_logging(log_dir, model_name):
    exp_name = f"{model_name}_{datetime.datetime.now().strftime('%m%d-%H%M%S')}"
    exp_path = Path(log_dir) / exp_name
    exp_path.mkdir(parents=True, exist_ok=True)
    ckpt_path = exp_path / "ckpt"
    ckpt_path.mkdir(exist_ok=True)
    loss_path = exp_path / "loss"
    loss_path.mkdir(exist_ok=True)
    return exp_path, ckpt_path, loss_path

def load_data(train_file, val_file, test_file, source_language, target_language):
    train_data = pd.read_csv(train_file, usecols=lambda col: col != 'Unnamed: 0')
    val_data = pd.read_csv(val_file, usecols=lambda col: col != 'Unnamed: 0')
    test_data = pd.read_csv(test_file, usecols=lambda col: col != 'Unnamed: 0')
    source_word_2_index = built_curpus(train_data[source_language])
    target_word_2_index = built_curpus(train_data[target_language])
    return train_data, val_data, test_data, source_word_2_index, target_word_2_index

def create_dataloaders(train_data, val_data, test_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length, batch_size):
    train_dataset = TranslationDataset(train_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
    val_dataset = TranslationDataset(val_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
    test_dataset = TranslationDataset(test_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, exp_path, ckpt_path, loss_path, device,max_grad_norm=1.0):
    log_file = exp_path / 'training_log.txt'
    loss_file = loss_path / 'losses.txt'

    with open(log_file, 'w') as log_f, open(loss_file, 'w') as loss_f:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for source_tokens, target_tokens in train_loader:
                source_tokens = source_tokens.to(device)  # Move data to device
                target_tokens = target_tokens.to(device)  # Move data to device
                
                optimizer.zero_grad()
                output = model(source_tokens)
                output = output.view(-1, output.size(-1))
                target_tokens = target_tokens.view(-1)
                loss = criterion(output, target_tokens)
                loss.backward()
                if isinstance(model, LSTMNetworkWithAttention):  # Check if the model is an instance of LSTMNetwork
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # Gradient clipping
                    
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            log_f.write(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}\n")
            loss_f.write(f"{epoch+1},{avg_train_loss}\n")
            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}")

            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for source_tokens, target_tokens in val_loader:
                    source_tokens = source_tokens.to(device)  # Move data to device
                    target_tokens = target_tokens.to(device)  # Move data to device
                    
                    output = model(source_tokens)
                    output = output.view(-1, output.size(-1))
                    target_tokens = target_tokens.view(-1)
                    loss = criterion(output, target_tokens)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            log_f.write(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}\n")
            loss_f.write(f"{epoch+1},{avg_val_loss}\n")
            print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss}")

            # Save checkpoint
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, ckpt_path / f'model_epoch_{epoch+1}.pth')
           
def evaluate_model(model, test_loader, criterion, device, exp_path):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for source_tokens, target_tokens in test_loader:
            source_tokens = source_tokens.to(device)  # Move data to device
            target_tokens = target_tokens.to(device)  # Move data to device
            
            output = model(source_tokens)
            output = output.view(-1, output.size(-1))
            target_tokens = target_tokens.view(-1)
            loss = criterion(output, target_tokens)
            test_loss += loss.item()
    avg_test_loss = test_loss / len(test_loader)
    print(f"Final Test Loss: {avg_test_loss}")
     # Save the test loss to a file
    test_loss_file = exp_path / 'test_loss.txt'
    with open(test_loss_file, 'w') as f:
        f.write(f"Final Test Loss: {avg_test_loss}\n")

if __name__ == "__main__":
    args = parse_args()
    source_language = 'indonesian'
    target_language = 'english'

    # Load data and build vocabularies
    train_data, val_data, test_data, source_word_2_index, target_word_2_index = load_data(
        args.train_file, args.val_file, args.test_file, source_language, target_language)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data, val_data, test_data, source_language, target_language,
        source_word_2_index, target_word_2_index, args.max_length, args.batch_size)

    # Initialize model
    vocab_size = len(source_word_2_index)
    num_classes = len(target_word_2_index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    if model_name == 'CNN':
        model = CNNModel(input_dim=vocab_size, embedding_dim=args.embedding_dim, num_classes=num_classes,dropout_prob=0.3)
    elif model_name == 'LSTM':
        hidden_dim = 128  # Example hidden dimension; you can adjust it
        num_layers = 2  # Example number of LSTM layers; you can adjust it
        bidirectional = True  # Change to True if you want a bidirectional LSTM

        model = LSTMNetworkWithAttention(
        input_dim=args.embedding_dim,  # The dimension of the input embeddings
        hidden_dim=hidden_dim,
        vocab_size=num_classes,  # Assuming num_classes corresponds to the output vocabulary size
        num_layers=num_layers,
        bidirectional=bidirectional,
        LSTM_drop=0.1,
        FC_drop=0.3)
    elif model_name == 'CNN_1':
        
        hidden_dim = 256
        model = TextCNNModel( vocab_size=num_classes, max_len=args.max_length, hidden_num=256, embedding_dim=512,dropout_rate=0.5)
    else:
        raise ValueError("Invalid model_name.")
    model = model.to(device)

    # Setup logging paths
    exp_path, ckpt_path, loss_path = setup_experiment_logging(args.log_dir, model_name)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate,weight_decay=1e-5)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.9, eps=1e-8, weight_decay=1e-5)


    # Train and evaluate
    train_model(model, train_loader, val_loader, criterion, optimizer, args.num_epochs, exp_path, ckpt_path, loss_path, device)
    evaluate_model(model, test_loader, criterion, device, exp_path)

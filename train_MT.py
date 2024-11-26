import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from Data_loader import TranslationDataset, built_curpus
from cnn_model import CNNModel
from lstm_model import LSTMNetwork, LSTMEncoder, LSTMDecoder
from rnn_model import RNNNetwork, RNNEncoder, RNNDecoder
import transformer 
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train Machine Translation Model")
    parser.add_argument('--source_language', type=str, default='indonesian', help='type of source language')
    parser.add_argument('--target_language', type=str, default='english', help='type of target language')
    parser.add_argument('--train_file', type=str, default='./nusax-main/datasets/mt/train.csv', help='Path to the training data file')
    parser.add_argument('--val_file', type=str, default='./nusax-main/datasets/mt/valid.csv', help='Path to the validation data file')
    parser.add_argument('--test_file', type=str, default='./nusax-main/datasets/mt/test.csv', help='Path to the test data file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=512, help='Dimension of embeddings')
    parser.add_argument('--num_epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate for optimizer')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum sequence length for input data')
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN','LSTM','RNN', 'transformer'], help='Type of model to use')
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


def create_dataloaders(model, train_data, val_data, test_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length, batch_size):
    if model == 'transformer':
        train_dataset = transformer.TranslationDataset_transformer(train_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        val_dataset = transformer.TranslationDataset_transformer(val_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        test_dataset = transformer.TranslationDataset_transformer(test_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = TranslationDataset(train_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        val_dataset = TranslationDataset(val_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        test_dataset = TranslationDataset(test_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def pretrain_transformer(args):

    exp_path, ckpt_path, loss_path = setup_experiment_logging(args.log_dir, args.model)

    train_data, val_data, test_data, source_word_2_index, target_word_2_index = load_data(
        args.train_file, args.val_file, args.test_file, args.source_language, args.target_language)
    
    extra_data_source = pd.read_csv('identic.csv', usecols=lambda col: col != 'Unnamed: 0')[args.source_language][:5000]
    extra_data_target = pd.read_csv('identic.csv', usecols=lambda col: col != 'Unnamed: 0')[args.target_language][:5000]

    total_data_source = pd.concat([train_data[args.source_language], extra_data_source], ignore_index=True)
    total_data_target = pd.concat([train_data[args.target_language], extra_data_target], ignore_index=True)

    # Build vocabularies
    source_word_2_index = built_curpus(total_data_source)
    target_word_2_index = built_curpus(total_data_target)

    dataset = transformer.TranslationDataset_transformer_extra(total_data_source, total_data_target, source_word_2_index, target_word_2_index, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    transformer.Train.src_language = args.source_language
    transformer.Train.tgt_language = args.target_language
    transformer.Train.source_word_2_index = source_word_2_index
    transformer.Train.target_word_2_index = target_word_2_index
    transformer.Train.src_vocab = len(source_word_2_index)
    transformer.Train.target_vocab = len(target_word_2_index)
    transformer.Train.N = 2
    transformer.Train.Fixed_len = args.max_length
    transformer.Train.Train_Data = val_data
    transformer.Train.Dataloader = dataloader
    transformer.Train.Epoch = args.num_epochs
    transformer.Train.Batch_size = args.batch_size
    transformer.Train.log_file_path = exp_path / 'training_log.txt'
    transformer.Train.loss_file_path = loss_path / 'losses.txt'
    transformer.Train.ckpt_file_path = ckpt_path / 'pretrain_transformer.pth'
    transformer.Train.test_file_path = './nusax-main/datasets/mt/valid.csv'

    transformer.train()

    test_file = './nusax-main/datasets/mt/test.csv'
    transformer.test(test_file)

def train_transformer(args):

    exp_path, ckpt_path, loss_path = setup_experiment_logging(args.log_dir, args.model)

    train_data, val_data, test_data, source_word_2_index, target_word_2_index = load_data(
        args.train_file, args.val_file, args.test_file, args.source_language, args.target_language)
    
    # data = pd.concat([train_data, test_data], ignore_index=True)
    data = train_data

    # Build vocabularies
    source_word_2_index = built_curpus(data[args.source_language])
    target_word_2_index = built_curpus(data[args.target_language])

    dataset = transformer.TranslationDataset_transformer(data, args.source_language, args.target_language, source_word_2_index, target_word_2_index, args.max_length)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    transformer.Train.src_language = args.source_language
    transformer.Train.tgt_language = args.target_language
    transformer.Train.source_word_2_index = source_word_2_index
    transformer.Train.target_word_2_index = target_word_2_index
    transformer.Train.src_vocab = len(source_word_2_index)
    transformer.Train.target_vocab = len(target_word_2_index)
    transformer.Train.N = 2
    transformer.Train.Fixed_len = args.max_length
    transformer.Train.Train_Data = val_data
    transformer.Train.Dataloader = dataloader
    transformer.Train.Epoch = args.num_epochs
    transformer.Train.Batch_size = args.batch_size
    transformer.Train.log_file_path = exp_path / 'training_log.txt'
    transformer.Train.loss_file_path = loss_path / 'losses.txt'
    transformer.Train.ckpt_file_path = ckpt_path / 'transformer.pth'
    transformer.Train.test_file_path = './nusax-main/datasets/mt/valid.csv'

    transformer.train()

    test_file = './nusax-main/datasets/mt/test.csv'
    transformer.test(test_file)

def train_nn(args, model, train_loader, val_loader, criterion, optimizer, num_epochs, exp_path, ckpt_path, loss_path, device):
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
                if args.model == "CNN":
                    output = model(source_tokens)
                else:
                    output = model(source_tokens, target_tokens)
                output = output.view(-1, output.size(-1))
                target_tokens = target_tokens.view(-1)
                loss = criterion(output, target_tokens)
                loss.backward()
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
                    
                    if args.model == "CNN":
                        output = model(source_tokens)
                    else:
                        output = model(source_tokens, target_tokens)
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
            
def test_transformer(args):
    exp_path, ckpt_path, loss_path = setup_experiment_logging(args.log_dir, args.model)

    train_data, val_data, test_data, source_word_2_index, target_word_2_index = load_data(
        args.train_file, args.val_file, args.test_file, args.source_language, args.target_language)
    
    data = pd.concat([train_data, test_data], ignore_index=True)

    source_word_2_index = built_curpus(data[args.source_language])
    target_word_2_index = built_curpus(data[args.target_language])

    transformer.Train.src_language = args.source_language
    transformer.Train.tgt_language = args.target_language
    transformer.Train.source_word_2_index = source_word_2_index
    transformer.Train.target_word_2_index = target_word_2_index
    transformer.Train.src_vocab = len(source_word_2_index)
    transformer.Train.target_vocab = len(target_word_2_index)
    transformer.Train.N = 2
    transformer.Train.ckpt_file_path = './ckpt/transformer.pth'
    transformer.Train.Fixed_len = args.max_length
    test_file = './nusax-main/datasets/mt/test.csv'
    transformer.test(test_file)


def test_nn(model, test_loader, criterion, device, exp_path):
    model.eval()
    test_loss = 0
    translations = []
    with torch.no_grad():
        for source_tokens, target_tokens in test_loader:
            source_tokens = source_tokens.to(device)  # Move data to device
            target_tokens = target_tokens.to(device)  # Move data to device
            if args.model == "CNN":
                output = model(source_tokens)
            else:
                output = model(source_tokens, target_tokens)
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
    
    model_name = args.model
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'transformer':
        pretrain_transformer(args)
    else:
        model = []
        train_data, val_data, test_data, source_word_2_index, target_word_2_index = load_data(
            args.train_file, args.val_file, args.test_file, args.source_language, args.target_language)
        train_loader, val_loader, test_loader = create_dataloaders(
            model, train_data, val_data, test_data, args.source_language, args.target_language,
            source_word_2_index, target_word_2_index, args.max_length, args.batch_size)
        vocab_size = len(source_word_2_index)
        num_classes = len(target_word_2_index)
        if model_name == 'CNN':
            model = CNNModel(input_dim=vocab_size, embedding_dim=args.embedding_dim,   
                             num_classes=num_classes, pad_token_id=0)
        elif model_name == 'LSTM':
            hidden_dim = 128  # Example hidden dimension; you can adjust it
            num_layers = 2  # Example number of LSTM layers; you can adjust it
            bidirectional = False  # Change to True if you want a bidirectional LSTM
            encoder = LSTMEncoder(input_dim=args.embedding_dim,  
                              hidden_dim=hidden_dim,
                              vocab_size=vocab_size, 
                              pad_token_id=0,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
            decoder = LSTMDecoder(output_dim=num_classes, 
                              hidden_dim=hidden_dim,
                              pad_token_id=0,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
            model = LSTMNetwork(encoder, decoder)
        elif model_name == 'RNN':
            hidden_dim = 256  # Example hidden dimension; you can adjust it
            num_layers = 5  # Example number of RNN layers; you can adjust it
            bidirectional = False  # Change to True if you want a bidirectional RNN

            encoder = RNNEncoder(input_dim=args.embedding_dim, 
                              hidden_dim=hidden_dim, 
                              vocab_size=vocab_size, 
                              pad_token_id=0,
                              num_layers=num_layers,       
                              bidirectional=bidirectional)
            decoder = RNNDecoder(output_dim=num_classes, 
                              hidden_dim=hidden_dim, 
                              pad_token_id=0,
                              num_layers=num_layers,
                              bidirectional=bidirectional)
            model = RNNNetwork(encoder, decoder)
        else:
            raise ValueError("Invalid model_name.")
        model = model.to(device)
        # Setup logging paths
        exp_path, ckpt_path, loss_path = setup_experiment_logging(args.log_dir, model_name)
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        # Train and evaluate
        train_nn(args, model, train_loader, val_loader, criterion, optimizer, args.num_epochs, exp_path, ckpt_path, loss_path, device)
        test_nn(args, model, test_loader, criterion, device, exp_path)

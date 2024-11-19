import os
import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from Data_loader import TranslationDataset, built_curpus
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
    parser.add_argument('--model', type=str, default='CNN', choices=['CNN','LSTM','CNN_1', 'transformer'], help='Type of model to use')
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
    elif model == 'model name':
        # to do
        pass
    else:
        train_dataset = TranslationDataset(train_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        val_dataset = TranslationDataset(val_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        test_dataset = TranslationDataset(test_data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def train_transformer(args):

    exp_path, ckpt_path, loss_path = setup_experiment_logging(args.log_dir, args.model)

    train_data, val_data, test_data, source_word_2_index, target_word_2_index = load_data(
        args.train_file, args.val_file, args.test_file, args.source_language, args.target_language)
    
    data = pd.concat([train_data, test_data], ignore_index=True)

    # Build vocabularies
    source_word_2_index = built_curpus(data[args.target_language])
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
    transformer.Train.Train_Data = data
    transformer.Train.Dataloader = dataloader
    transformer.Train.Epoch = args.num_epochs
    transformer.Train.Batch_size = args.batch_size
    transformer.Train.log_file_path = exp_path / 'training_log.txt'
    transformer.Train.loss_file_path = loss_path / 'losses.txt'
    transformer.Train.ckpt_file_path = ckpt_path / 'transformer.pth'

    transformer.train()


def train_LSTM(args):
    pass


def train_CNN(args):
    pass


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
    test_file = './nusax-main/datasets/mt/valid.csv'
    transformer.test(test_file)


def test_LSTM(args):
    pass

def test_CNN(args):
    pass

if __name__ == "__main__":

    args = parse_args()
    
    model_name = args.model

    if model_name == 'transformer':
        if args.mode == 'train':
            train_transformer(args)
        elif args.mode == 'test':
            test_transformer(args)
    elif model_name == 'LSTM':
        if args.mode == 'train':
            train_LSTM(args)
        elif args.mode == 'test':
            test_LSTM(args)
    elif model_name == 'CNN':
        if args.mode == 'train':
            train_CNN(args)
        elif args.mode == 'test':
            test_CNN(args)
    else:
        raise ValueError("Invalid model_name.")


import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer

class TranslationDataset(Dataset):
    def __init__(self, file_path, source_language, target_language, tokenizer):
        # Load the dataset
        self.data = pd.read_csv(file_path, usecols=lambda col: col != 'Unnamed: 0')
        # Specify the columns for source and target languages
        self.source_texts = self.data[source_language].tolist()
        self.target_texts = self.data[target_language].tolist()
        self.tokenizer = tokenizer  # BERT tokenizer instance

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        # Retrieve the source and target sentences
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]
        
        # Tokenize the source and target texts using the BERT tokenizer
        source_tokens = self.tokenizer.encode(source_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)[0]
        target_tokens = self.tokenizer.encode(target_text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)[0]
        
        return source_tokens, target_tokens

# Custom collate function for padding (if sequences need to be adjusted manually)
def collate_fn(batch):
    # Separate the source and target sequences
    sources, targets = zip(*batch)
    
    # Pad sequences to the same length using PyTorch's pad_sequence
    sources_padded = pad_sequence(sources, batch_first=True, padding_value=0)
    targets_padded = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return sources_padded, targets_padded

# Usage example
file_path = './nusax-main/datasets/mt/train.csv'
source_language = 'indonesian'  # Specify source language column
target_language = 'english'  # Specify target language column

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the dataset instance
dataset = TranslationDataset(file_path, source_language, target_language, tokenizer)

# Create DataLoader with the custom collate function
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Example iteration over the DataLoader
for source, target in dataloader:
    print("Source:", source)
    print("Target:", target)
    break

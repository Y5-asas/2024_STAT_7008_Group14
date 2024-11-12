import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
# Function to build a word-to-index mapping (vocabulary)
def built_curpus(train_texts):
    word_2_index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for text in train_texts:
        for word in text.split():
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index

# TranslationDataset class
class TranslationDataset(Dataset):
    def __init__(self, data, source_language, target_language, source_word_2_index, target_word_2_index, max_length=1000):
        """
        Initializes the TranslationDataset.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the translation data.
        - source_language (str): Column name for the source language.
        - target_language (str): Column name for the target language.
        - source_word_2_index (dict): Word-to-index mapping for the source language.
        - target_word_2_index (dict): Word-to-index mapping for the target language.
        - max_length (int): Maximum sequence length for padding/truncation.
        """
        self.source_texts = data[source_language].tolist()
        self.target_texts = data[target_language].tolist()
        self.source_word_2_index = source_word_2_index
        self.target_word_2_index = target_word_2_index
        self.max_length = max_length

    def tokenize_and_pad(self, text, word_2_index):
        """
        Tokenizes and pads a single text.

        Parameters:
        - text (str): The input text (sentence).
        - word_2_index (dict): Word-to-index mapping.

        Returns:
        - torch.Tensor: Tokenized and padded tensor.
        """
        # Tokenize words to indices
        tokenized = [word_2_index.get(word, word_2_index["<UNK>"]) for word in text.split()]
        # Add <SOS> and <EOS> tokens
        tokenized = [word_2_index["<SOS>"]] + tokenized + [word_2_index["<EOS>"]]
        # Pad or truncate to max_length
        if len(tokenized) < self.max_length:
            tokenized += [word_2_index["<PAD>"]] * (self.max_length - len(tokenized))
        else:
            tokenized = tokenized[:self.max_length]
        return torch.tensor(tokenized, dtype=torch.long)

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]

        # Tokenize and pad the source and target texts
        source_tokens = self.tokenize_and_pad(source_text, self.source_word_2_index)
        target_tokens = self.tokenize_and_pad(target_text, self.target_word_2_index)

        return source_tokens, target_tokens

# Load data (example usage)
if __name__ == "__main__":
    file_path = './nusax-main/datasets/mt/train.csv'
    data = pd.read_csv(file_path, usecols=lambda col: col != 'Unnamed: 0')
    source_language = 'indonesian'
    target_language = 'english'

    # Build vocabularies
    source_word_2_index = built_curpus(data[source_language])
    target_word_2_index = built_curpus(data[target_language])
    print(len(source_word_2_index)) # Check whether the index list is correctly formed 

    # Create Dataset and DataLoader
    max_length = 1000
    dataset = TranslationDataset(data, source_language, target_language, source_word_2_index, target_word_2_index, max_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    embedding_dim = 512

    # Example usage
    for batch_idx, (source_tokens, target_tokens) in enumerate(dataloader):
        source_emb = nn.Embedding(len(source_word_2_index),embedding_dim)
        
        source_tokens = source_emb(source_tokens)
        
        print(f"Batch {batch_idx + 1}")
        print("Source Tokens Shape:", source_tokens.shape)
        print("Target Tokens Shape:", target_tokens.shape)
        break  # Only show the first batch for demonstration

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder


# Function to build a word-to-index mapping (vocabulary)
def built_curpus(train_texts):
    word_2_index = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
    for text in train_texts:
        for word in text.split():
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index

class SentiDataset(Dataset):
    def __init__(self, data, senti_word_2_index, max_length=1000):
        """
        Parameters:
        - data: The senti dataset
        - senti_word_2_index (dict): Word-to-index mapping for the text.
        - max_length (int): Maximum sequence length for padding/truncation.
        """
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(data['label'].tolist())
        # labels = [[float(int(label == 0)), float(int(label == 1)), float(int(label == 2))] for label in labels]

        self.labels = torch.tensor(labels, dtype=torch.long)
        self.texts = data['text'].tolist()
        self.senti_word_2_index = senti_word_2_index
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
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize and pad the texts
        tokens = self.tokenize_and_pad(text, self.senti_word_2_index)

        return tokens, label


# Load data (example usage)
if __name__ == "__main__":
    file_path = 'nusax-main/datasets/sentiment/indonesian/train.csv'
    data = pd.read_csv(file_path, usecols=lambda col: col != 'Unnamed: 0')

    # Build vocabularies
    senti_word_2_index = built_curpus(data['text'])

    # Create Dataset and DataLoader
    max_length = 1000
    dataset = SentiDataset(data, senti_word_2_index, max_length)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    embedding_dim = 512

    # Example usage
    for batch_idx, (tokens, label) in enumerate(dataloader):
        source_emb = nn.Embedding(len(senti_word_2_index), embedding_dim)

        tokens = source_emb(tokens)

        print(f"Batch {batch_idx + 1}")
        print("Source Tokens Shape:", tokens.shape)
        print(label)
        break  # Only show the first batch for demonstration

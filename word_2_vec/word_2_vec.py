import gensim
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader


# TranslationDataset class
class TranslationDataset(Dataset):
    def __init__(self, data, source_language, target_language, word2vec_model, max_length=1000):
        """
        Initializes the TranslationDataset.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the translation data.
        - source_language (str): Column name for the source language.
        - target_language (str): Column name for the target language.
        - word2vec_model: Pre-trained Word2Vec model.
        - max_length (int): Maximum sequence length for padding/truncation.
        """
        self.source_texts = data[source_language].tolist()
        self.target_texts = data[target_language].tolist()
        self.word2vec_model = word2vec_model
        self.max_length = max_length

    def tokenize_and_pad(self, text):
        """
        Tokenizes and pads a single text using Word2Vec embeddings.

        Parameters:
        - text (str): The input text (sentence).

        Returns:
        - torch.Tensor: Tokenized and padded tensor of embeddings.
        """
        embeddings = []
        for word in text.split():
            if word in self.word2vec_model:
                embeddings.append(torch.tensor(self.word2vec_model[word]))
            else:
                embeddings.append(torch.zeros(self.word2vec_model.vector_size))  # Handle OOV words

        # Pad or truncate to max_length
        if len(embeddings) < self.max_length:
            padding = [torch.zeros(self.word2vec_model.vector_size)] * (self.max_length - len(embeddings))
            embeddings.extend(padding)
        else:
            embeddings = embeddings[:self.max_length]

        return torch.stack(embeddings)

    def __len__(self):
        return len(self.source_texts)

    def __getitem__(self, idx):
        source_text = self.source_texts[idx]
        target_text = self.target_texts[idx]

        source_tokens = self.tokenize_and_pad(source_text)
        target_tokens = self.tokenize_and_pad(target_text)

        return source_tokens, target_tokens


# Example usage
if __name__ == "__main__":
    word2vec_path = "./cc.id.300.vec/cc.id.300.vec"  # Pretrained Word2Vec text file
    print(f"Loading Word2Vec model from {word2vec_path}...")
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False)
    print("Word2Vec model loaded!")

    file_path = './nusax-main/datasets/mt/train.csv'
    data = pd.read_csv(file_path, usecols=lambda col: col != 'Unnamed: 0')
    source_language = 'indonesian'
    target_language = 'english'

    max_length = 1000
    dataset = TranslationDataset(data, source_language, target_language, word2vec, max_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for batch_idx, (source_tokens, target_tokens) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")
        print("Source Tokens Shape:", source_tokens.shape)  # (batch_size, max_length, embedding_dim)
        print("Target Tokens Shape:", target_tokens.shape)  # (batch_size, max_length, embedding_dim)
        break  # Only show the first batch for demonstration

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from Data_loader import built_curpus
from sentimodel import MyTransformerModel

# SentimentDataset class
class SentimentDataset_transformer(Dataset):
    def __init__(self, data, text_column, label_column, word_2_index, max_length=1000):
        """
        Initializes the SentimentDataset.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing the sentiment data.
        - text_column (str): Column name for the text data.
        - label_column (str): Column name for the labels.
        - word_2_index (dict): Word-to-index mapping for the text data.
        - max_length (int): Maximum sequence length for padding/truncation.
        """
        self.texts = data[text_column].tolist()
        self.labels = data[label_column].tolist()
        self.word_2_index = word_2_index
        self.max_length = max_length
        self.target_map = {'positive':1, 'negative':0, 'neutral':2}

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
        # Add <EOS> token
        tokenized = tokenized + [word_2_index["<EOS>"]]
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
        label = self.target_map[self.labels[idx]]

        # Tokenize and pad the text
        text_tokens = self.tokenize_and_pad(text, self.word_2_index)
        return text_tokens, torch.tensor(label, dtype=torch.long)

def load_data(train_file, valid_file, test_file):
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    test_data = pd.read_csv(test_file)
    word_2_index = built_curpus(train_data['text'])
    return train_data, valid_data, test_data, word_2_index

train_path = './nusax-main/datasets/sentiment/indonesian/train.csv'
valid_path = './nusax-main/datasets/sentiment/indonesian/valid.csv'
test_path = './nusax-main/datasets/sentiment/indonesian/test.csv'
batch_size = 32
max_length = 100
learning_rate = 2e-3
max_epochs = 30



train_data, valid_data, test_data, word_2_index = load_data(train_path, valid_path, test_path)


dataset = SentimentDataset_transformer(train_data, 'text', 'label', word_2_index, max_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MyTransformerModel(vocab_size=len(word_2_index), embedding_dim=512, p_drop=0.1, h=8, output_size=3)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index = 0)
model.train()
for epoch in range(max_epochs):
    total_loss = 0
    for i, (text, label) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(text, None)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss}")



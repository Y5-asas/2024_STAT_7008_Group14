import torch
import torch.nn as nn

class LSTMNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, num_layers=1, bidirectional=False):
        super(LSTMNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim=input_dim)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1  # Bidirectional LSTM doubles the number of directions

        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layer to map LSTM output to the vocabulary size
        self.fc = nn.Linear(hidden_dim * self.num_directions, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)  # Get the batch size dynamically
        
        # Initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)

        # Forward propagate the LSTM
        out, _ = self.lstm(x, (h0, c0))  # `out` shape: [batch_size, q_len, hidden_dim * num_directions]

        # Pass the LSTM output through the fully connected layer
        out = self.fc(out)  # `out` shape: [batch_size, q_len, vocab_size]
        return out

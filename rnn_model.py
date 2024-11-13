import torch
import torch.nn as nn

class RNNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, pad_token_id, num_layers=1, bidirectional=False):
        super(RNNNetwork, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=input_dim, padding_idx=pad_token_id)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1  # Bidirectional RNN doubles the number of directions

        # Define the RNN layer
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        # Fully connected layer to map RNN output to the vocabulary size
        self.fc = nn.Linear(hidden_dim * self.num_directions, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)  # Get the batch size dynamically
        
        # Initialize hidden state (h0) with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)

        # Forward propagate the RNN
        out, _ = self.rnn(x, h0)  # `out` shape: [batch_size, seq_len, hidden_dim * num_directions]

        # Pass the RNN output through the fully connected layer
        out = self.fc(out)  # `out` shape: [batch_size, seq_len, vocab_size]
        return out

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, pad_token_id, num_layers=1, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=input_dim, padding_idx=pad_token_id)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1 
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)

        # Initialize hidden state (h0) and cell state (c0) with zeros
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)

        # Forward propagate the LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        return hn, cn  # Return the last hidden state and cell state for the decoder

class LSTMDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, pad_token_id, num_layers=1, bidirectional=False):
        super(LSTMDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=pad_token_id)
        self.num_directions = 2 if bidirectional else 1 
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)  # Output shape: [batch_size, seq_len, output_dim]
        return out, hidden

class LSTMNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super(LSTMNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src: source sequence, trg: target sequence
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.fc.out_features

        # Initialize the outputs tensor
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(src.device)

        # Get the hidden state and cell state from the encoder
        hidden, cell = self.encoder(src)

        # Prepare the first input to the decoder (start token)
        input_dec = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, (hidden, cell) = self.decoder(input_dec, (hidden, cell))
            outputs[:, t] = output.squeeze(1)
            input_dec = output.argmax(dim=2)  # Use the predicted output as the next input

        return outputs

import torch
import torch.nn as nn

class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_size, pad_token_id, num_layers=1, bidirectional=False):
        super(RNNEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=input_dim, padding_idx=pad_token_id)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1 
        
        self.rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x):
        x = self.embedding(x)
        batch_size = x.size(0)

        # Initialize hidden state (h0) with zeros
        h0 = torch.zeros(self.rnn.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)

        # Forward propagate the RNN
        out, hn = self.rnn(x, h0)
        return hn  # Return the last hidden state for the decoder

class RNNDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, pad_token_id, num_layers=1, bidirectional=False):
        super(RNNDecoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim, padding_idx=pad_token_id)
        self.num_directions = 2 if bidirectional else 1 
        self.rnn = nn.RNN(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * self.num_directions, output_dim)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out)  # Output shape: [batch_size, seq_len, output_dim]
        return out, hidden

class RNNNetwork(nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        # src: source sequence, trg: target sequence
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        output_dim = self.decoder.fc.out_features

        # Initialize the outputs tensor
        outputs = torch.zeros(batch_size, trg_len, output_dim).to(src.device)

        # Get the hidden state from the encoder
        hidden = self.encoder(src)

        # Prepare the first input to the decoder (start token)
        input_dec = trg[:, 0].unsqueeze(1)

        for t in range(1, trg_len):
            output, hidden = self.decoder(input_dec, hidden)
            outputs[:, t] = output.squeeze(1)
            input_dec = output.argmax(dim=2)

        return outputs

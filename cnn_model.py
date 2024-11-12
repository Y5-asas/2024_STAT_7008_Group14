import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_classes, kernel_sizes=[3, 4, 5], num_filters=100):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        
        # Convolutional layers with padding
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (kernel_size, embedding_dim), padding=(kernel_size // 2, 0)) for kernel_size in kernel_sizes
        ])
        
        # Fully connected layer to map to num_classes (vocab size)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        # Embed input
        embedded = self.embedding(x).unsqueeze(1)  # [batch_size, 1, seq_len, embedding_dim]
        
        # Apply convolutions
        conv_results = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # List of [batch_size, num_filters, seq_len]
        # Find the minimum length among all conv results
        min_length = min(conv_result.size(2) for conv_result in conv_results)

        # Trim all conv results to the minimum length
        conv_results = [conv_result[:, :, :min_length] for conv_result in conv_results]

        # Concatenate results
        cat = torch.cat(conv_results, dim=1)
        
        cat = torch.cat(conv_results, dim=1)  # Shape: [batch_size, num_filters * len(kernel_sizes), seq_len]
        
        # Apply a linear layer across the sequence dimension
        cat = cat.permute(0, 2, 1)  # Reshape to [batch_size, seq_len, num_filters * len(kernel_sizes)]
        output = self.fc(cat)  # Shape: [batch_size, seq_len, num_classes]
        
        return output


import torch
import torch.nn as nn

import torch.nn.functional as F



class CNNnet(nn.Module):
    def __init__(self, input_size, embedding_dim, num_classes, dropout = 0.2, kernel_num = 100, kernel_sizes = [3, 4, 5]):
        super().__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim,padding_idx=0)
        self.conv11 = nn.Conv2d(1, kernel_num, (kernel_sizes[0], embedding_dim))
        self.conv12 = nn.Conv2d(1, kernel_num, (kernel_sizes[1], embedding_dim))
        self.conv13 = nn.Conv2d(1, kernel_num, (kernel_sizes[2], embedding_dim))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(len(kernel_sizes) * kernel_num, num_classes)

    @staticmethod
    def conv_and_pool(x, conv):
        # x: (batch, 1, sequence_length, )
        x = conv(x)
        # x: (batch, kernel_num, out, 1)
        x = F.relu(x.squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        # x: (batch, kernel_num)
        return x

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        x1 = self.conv_and_pool(x, self.conv11)
        x2 = self.conv_and_pool(x, self.conv12)
        x3 = self.conv_and_pool(x, self.conv13)

        x = torch.cat((x1, x2, x3), 1)

        x = self.dropout(x)

        x = self.fc1(x)
        return x


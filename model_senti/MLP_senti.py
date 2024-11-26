
import torch
import torch.nn as nn




class MLP(nn.Module):
    '''
    naive fc layer
    '''

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim),
                                 nn.BatchNorm1d(out_dim),
                                 nn.Dropout(dropout),
                                 nn.ReLU(),
                                 )

    def forward(self, x):
        x = self.mlp(x)
        return x


class MLPModel(nn.Module):
    def __init__(self, input_size, embedding_dim, num_classes, dropout, hidden_layers = [1024, 512, 10]):
        super(MLPModel, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim,padding_idx=0)
        self.fc0 = MLP(embedding_dim * 1000, hidden_layers[0], dropout)
        self.fcs = nn.ModuleList([MLP(hidden_layers[i - 1], hidden_layers[i], dropout) for i in range(1, len(hidden_layers))])
        self.fc1 = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.flatten(x, 1)
        x = self.fc0(x)
        for fc in self.fcs:
            x = fc(x)
        x = self.fc1(x)
        return x



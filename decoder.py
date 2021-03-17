import torch
from Inception import inception_v3

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, embed_size, num_layers, feature_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        # input = input.unsqueeze(0)

        output = self.embedding(input)

        output = F.relu(output)

        output, hidden = self.gru(output, hidden)

        output = self.softmax(self.out(output[0]))

        return output, hidden



import torch
import torch.nn as nn

import numpy as np


class Decoder(nn.Module):
    def __init__(
        self,
        in_feature_dim,
        conv_kernel,
        conv_stride,
        hidden_dim,
        num_layers,
        vocab_size,
        dropout=0,
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            in_feature_dim, hidden_dim, kernel_size=conv_kernel, stride=conv_stride
        )

        self.dropout = nn.Dropout(dropout)

        self.bidirectional_rnn = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # 42 phonemes and ctc blank token
        self.linear_projection = nn.Linear(hidden_dim * 2, vocab_size)

        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        output = self.conv(x)

        output = output.permute(0, 2, 1)

        output = self.dropout(output)

        output, hidden = self.bidirectional_rnn(output)

        output = self.linear_projection(output)

        return self.log_softmax(output)

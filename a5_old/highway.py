#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Highway(nn.Module):
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1f
    """
    Highway Layer for combining outputs of Conv1D and skip connection
    """
    def __init__(self, word_emb_size):
        """
        Highway layer initialization, set two linear layers with bias.

        :param word_emb_size (int): word embedding size, i.e. e_{word}
        """
        super(Highway, self).__init__()
        self.word_emb_size = word_emb_size
        self.proj = nn.Linear(in_features=self.word_emb_size,
                              out_features=self.word_emb_size,
                              bias=True)
        self.gate = nn.Linear(in_features=self.word_emb_size,
                              out_features=self.word_emb_size,
                              bias=True)

    def forward(self, X_conv_out):
        """
        Highway connection, for formula (8), (9), (10)

        :param X_conv_out (torch.Tensor): output of conv1D layer, shape=[batch_size, emb_size]
        :return: X_highway (torch.Tensor): output of Highway layer, shape=[batch_size, emb_size]
        """
        X_proj = F.relu(self.proj(X_conv_out))
        X_gate = torch.sigmoid(self.gate(X_conv_out))

        X_highway = X_gate.mul_(X_proj) + (1 - X_gate).mul_(X_conv_out)
        return X_highway


    ### END YOUR CODE

if __name__ == "__main__":
    pass
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2020-21: Homework 5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    1D Conv -> ReLU -> max pool module
    """
    # Remember to delete the above 'pass' after your implementation
    ### YOUR CODE HERE for part 1g
    def __init__(self, char_emb_size, word_emb_size, max_len, kernel_size=5, padding=1):
        """

        :param char_emb_size int: w_{char}
        :param word_emb_size int: w_{word}, i.e. the number of filters
        :param max_len int: maximum length of words
        :param kernel_size int: window size, default 5
        :param padding int:  padding scheme, default 1
        """
        super(CNN, self).__init__()

        self.conv = nn.Conv1d(in_channels=char_emb_size,
                              out_channels=word_emb_size,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=True)
        self.relu = nn.ReLU()

    def forward(self, X_reshaped):
        """
        Convolution connection, foumula (5), (6), (7)

        :param X_reshaped (torch.Tensor): input from character embeddings, shape=[batch_size, char_emb_size, max_len]
        :return X_convout (torch.Tensor): output of CNN module, shape=[batch_size, word_emb_size]
        """
        X_conv = self.relu(self.conv(X_reshaped))
        X_convout = torch.max(X_conv, dim=2)[0]

        return X_convout

    ### END YOUR CODE
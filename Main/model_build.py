#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SpectralCRNN_Reg_Dropout(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_Dropout, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1,3)),
            nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 1)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1,3)),
            nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3, 1)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1,3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3, 1))
        )
        self.rnn = nn.GRU(128, 200, batch_first = True)
        self.fc = nn.Linear(200, 1)
    def forward(self, x):
        print("Input:", x.shape)
        out = self.conv(x)  # [17661, 128, 14, 1]
        print("After conv1:", out.shape)
        out = out.squeeze(-1)  # Remove width dimension, now [17661, 128, 14]
        out = out.permute(0, 2, 1)  # Change to [17661, 14, 128] to match RNN input expectations
        out, _ = self.rnn(out, self.hidden)
        out = out[:, -1, :]  # Take last time step's output
        out = self.fc(out)
        return F.relu(out)
    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()

# Use the forward_pass function in your script
# Assume x is the input tensor (mel spectrogram images)
# output = forward_pass(x)

import torch.nn as nn

class SpectralCRNN(nn.Module):
    def __init__(self):
        super(SpectralCRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),  # Adjust kernel size and padding
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),  # Safe pooling operation
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),  # Adjust kernel size and padding
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),  # Further safe pooling
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),  # Ensure no zero-dimension output
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((2, 2)),  # Final pooling before RNN
        )
        self.rnn = nn.GRU(128, 200, batch_first=True)  # No change here
        self.fc = nn.Linear(200, 1)  # No change here

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(2)  # Removing an unnecessary dimension
        x = x.permute(0, 2, 1)  # Adjust dimensions for RNN input
        _, hidden = self.rnn(x)  # Getting the last hidden state
        x = self.fc(hidden[-1])  # Using the last output of the RNN
        return x

    def init_hidden(self, batch_size):
        hidden = torch.zeros(1, batch_size, 200)
        return hidden.cuda() if torch.cuda.is_available() else hidden

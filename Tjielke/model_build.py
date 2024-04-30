#!/usr/bin/env python
# coding: utf-8

# In[5]:


import torch
import torch.nn as nn
import torch.nn.functional as F

# Initialize the convolutional block
conv_block = nn.Sequential(
    # Conv Layer 1
    nn.Conv2d(1, 48, kernel_size=(3, 7), padding=(1,3)),
    nn.BatchNorm2d(48),
    nn.ELU(),
    nn.MaxPool2d((2,4)),

    # Conv Layer 2
    nn.Conv2d(48, 64, kernel_size=(3, 7), padding=(1,3)),
    nn.BatchNorm2d(64),
    nn.ELU(),
    nn.MaxPool2d((3,5)),

    # Conv Layer 3
    nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1,3)),
    nn.BatchNorm2d(128),
    nn.ELU(),
    nn.MaxPool2d((3,5))
)

# Initialize the GRU layer
input_size = 640  # This should be calculated based on the output shape of the convolutional block
hidden_size = 200
gru_layer = nn.GRU(input_size, hidden_size, batch_first=True)

# Initialize the fully connected layer
fc_layer = nn.Linear(hidden_size, 1)

# Define a function to initialize the hidden state of the GRU layer
def init_hidden(batch_size):
    hidden = torch.zeros(1, batch_size, hidden_size)
    if torch.cuda.is_available():
        hidden = hidden.cuda()
    return hidden

# Define the forward pass function
def forward_pass(x):
    # Pass through convolutional block
    out = conv_block(x)
    # Reshape and transpose the output
    out = out.view(out.size(0), -1, out.size(3))
    out = out.transpose(1, 2)

    # Initialize hidden state
    batch_size = x.size(0)
    hidden = init_hidden(batch_size)

    # Pass through GRU layer
    out, _ = gru_layer(out, hidden)

    # Get the last time step output
    out = out[:, -1, :]

    # Pass through fully connected layer
    out = fc_layer(out)

    return out

# Use the forward_pass function in your script
# Assume x is the input tensor (mel spectrogram images)
# output = forward_pass(x)

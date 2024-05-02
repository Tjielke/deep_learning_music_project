import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MusicModel(nn.Module):
    def __init__(self, dropout_rate = 0.2):
        super(MusicModel, self).__init__()
        self.dropout_rate = dropout_rate
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1: 32x32x3 -> 16x16x32
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2: 16x16x32 -> 8x8x64
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=self.dropout_rate),
            nn.Linear(8 * 8 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.fc_layer(x)
        return x

class SpectralCRNN_Reg_Dropout(nn.Module):
    def __init__(self):
        super(SpectralCRNN_Reg_Dropout, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 32, kernel_size=(3, 7), padding=(1,3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((2,4)),
            # Conv Layer 2
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1,3)),
            #nn.Dropout2d(0.6),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3,5)),
            # Conv Layer 3
            nn.Conv2d(64, 128, kernel_size=(3, 7), padding=(1,3)),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d((3,5))
        )
        self.rnn = nn.GRU(640, 200, batch_first = True)
        self.fc = nn.Linear(200, 1)
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out = out.transpose(1,2)
        out, _ = self.rnn(out, self.hidden)
        out = out[:,-1,:]
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
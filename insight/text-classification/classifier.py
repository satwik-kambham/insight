import torch
import torch.nn as nn
import torch.optim as optim

import pytorch_lightning as pl


class Classifier(pl.LightningModule):
    def __init__(self, vocab_size, hidden_size=30, num_layers=1, lr=0.0001):
        super().__init__()
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.lr = lr

        self.rnn = nn.RNN(self.vocab_size, self.hidden_size, self.num_layers)
        self.head = nn.Sequential(nn.LazyLinear(6))

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input):
        # input shape => (batch_size, vocab_size, tokens_in_sentence)
        # rnn accepts input of shape => (tokens_in_sentence, batch_size, vocab_size)
        batch_size = input.shape[0]
        input = input.permute(1, 0, 2)

        # D = 1 for unidirectional and 2 for bi-directional
        # hidden state shape => (D * num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device)
        print(input.dtype, h0.dtype)
        out, hn = self.rnn(input, h0)
        # output shape => (tokens_in_sentence, batch_size, D * hidden_size)
        # hidden_shape => (D * num_layers, batch_size, hidden_size)
        # use linear layer to convert output from hidden_shape to desized shape
        hn = hn.permute(1, 0, 2).flatten(start_dim=1)
        cls = self.head(hn)
        return cls

    def training_step(self, batch, batch_idx):
        input, coarse, fine = batch

        cls = self.forward(input)

        loss = self.criterion(cls, coarse)
        self.log("train_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        input, coarse, fine = batch

        cls = self.forward(input)

        return cls, coarse

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

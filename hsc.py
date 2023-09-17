import torch
from torch.optim import lr_scheduler

from torch import nn


class HSCNet(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.layers = layers.to(device)

    def forward(self, x):
        return self.layers(x)


# Trainer can train models for either soft-boundary Deep SVDD or One-Class Deep SVDD
class HSCTrainer:
    """
    Model that trains on HSC loss. Is run in semi supervised mode, with labeled data.
    """
    def __init__(self, model, learning_rate, epochs, batch_size, device):
        self.model = model
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

    def train(self, training_data, labels):
        train_loader = torch.utils.data.DataLoader(list(zip(training_data, labels)), batch_size=self.batch_size)

        for epoch in range(self.epochs):
            for data in train_loader:
                self.training_step(data, epoch)
            # plot(validation_data, self, epoch)

    def training_step(self, data, epoch):
        inputs, labels = data
        self.optimizer.zero_grad()
        outs = self.model.forward(inputs)
        loss = outs ** 2
        loss = (loss + 1).sqrt() - 1
        loss = loss.reshape(labels.size(0), -1).mean(-1)
        norm = loss[labels == 0]
        anom = (-(((1 - (-loss[labels == 1]).exp()) + 1e-31).log()))
        loss[(1 - labels).nonzero().squeeze()] = norm
        loss[labels.nonzero().squeeze()] = anom

        loss = loss.sum()
        loss.backward()
        self.optimizer.step()

        return loss.item()
        # utils.plot_hsc(outs, labels, epoch)

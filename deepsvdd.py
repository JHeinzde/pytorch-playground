import torch
import wandb
from torch import nn
from utils import norm
from numpy import sqrt, quantile


class DeepSVDD(nn.Module):
    def __init__(self, layers, autoencoder, device):
        super().__init__()
        self.layers = layers.to(device)
        net_dict = self.state_dict()
        ae_dict = autoencoder.state_dict()
        ae_dict = {k: v for k, v in ae_dict.items() if k in net_dict}
        net_dict.update(ae_dict)
        self.load_state_dict(net_dict)

    def forward(self, x):
        return self.layers(x)


# Trainer can train models for either soft-boundary Deep SVDD or One-Class Deep SVDD
class DeepSVDDTrainer:
    def __init__(self, model, goal, c_size, learning_rate, epochs, device):
        self.model = model
        self.goal = goal
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.nu = 0.1
        self.c = torch.tensor([0] * c_size, dtype=torch.float32, device=device)
        self.R = torch.tensor(0, dtype=torch.float32, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, training_data):
        train_loader = torch.utils.data.DataLoader(list(zip(training_data, training_data)), batch_size=1)
        warmup_epoch = 5
        losses = []

        for epoch in range(self.epochs):
            distances = []
            for data in train_loader:
                dist, loss = self.training_step(data)
                distances.append(dist)
            if epoch >= warmup_epoch and epoch % warmup_epoch == 0:
                self.R = torch.tensor(quantile(sqrt(distances), 1 - self.nu))
                wandb.log({'radius': self.R})
            wandb.log({'loss_dsvdd': loss })
        return losses

    def training_step(self, data):
        inputs, _ = data
        self.optimizer.zero_grad()
        outputs = self.model.forward(norm(inputs))
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        if self.goal == 'one-class':
            loss = torch.mean(dist)
        else:
            scores = dist - self.R ** 2
            # Use default nu of 0.1 for now
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        loss.backward()
        self.optimizer.step()
        return dist.item(), loss.item()

    def set_center(self, transformed_data):
        with torch.no_grad():
            for x in transformed_data:
                self.c += self.model.forward(torch.nn.functional.normalize(x, p=2, dim=0))

        self.c /= len(transformed_data)

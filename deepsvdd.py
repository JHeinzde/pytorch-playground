import torch
from torch.optim import lr_scheduler

import wandb
from torch import nn
from utils import norm, plot
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
        self.c_size = c_size
        self.c = torch.tensor([0] * c_size, dtype=torch.float32, device=device)
        self.R = torch.tensor(1, dtype=torch.float32, device=device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-6)
        #self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=2000, gamma=0.8)

    def train(self, training_data):
        train_loader = torch.utils.data.DataLoader(list(zip(training_data, training_data)), batch_size=50)

        for epoch in range(self.epochs):
            points = []
            for data in train_loader:
                self.training_step(data, epoch)
                points.append(data[0])
            plot(points, self, epoch)

    def training_step(self, data, epoch):
        warmup_epoch = 10
        inputs, _ = data
        self.optimizer.zero_grad()
        outputs = self.model.forward(norm(inputs))
        dist = torch.sum((outputs - self.c) ** 2, dim=1)
        if self.goal == 'one-class':
            loss = torch.mean(dist)
        else:
            scores = dist - self.R ** 2
            loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        loss.backward()
        self.optimizer.step()
        wandb.log({'loss_dsvdd': loss.item()})

        if epoch >= warmup_epoch:
            #self.R = torch.tensor(quantile(sqrt(dist.clone().data.cpu().numpy()), 1 - self.nu))
            #wandb.log({'radius': self.R})
            pass

    def set_center(self, trainins_data):
        eps = 1
        with torch.no_grad():
            self.c = torch.zeros(self.c_size, device='cuda')
            for x in trainins_data:
                self.c += self.model.forward(torch.nn.functional.normalize(x, p=2, dim=0))

            self.c /= len(trainins_data)
            self.c[(abs(self.c) < eps) & (self.c < 0)] = -eps
            self.c[(abs(self.c) < eps) & (self.c > 0)] = eps


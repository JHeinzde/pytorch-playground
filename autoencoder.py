import torch
from torch import nn
from utils import norm
from torch.optim import lr_scheduler
import wandb
from sklearn import preprocessing


class AutoEncoder(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.layers = layers.to(device)

    def forward(self, x):
        return self.layers(x)


class AETrainer:
    def __init__(self, model, device, epochs, learning_rate, batch_size):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.loss_function = nn.L1Loss(reduction='mean').to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.1)

    def train(self, training_data, validation_data, a_normal):
        train_loader = torch.utils.data.DataLoader(list(zip(training_data, training_data)), batch_size=self.batch_size,
                                                   shuffle=True)
        losses = []
        for _ in range(0, self.epochs):
            epoch_loss = []
            for i, data in enumerate(train_loader, 0):
                loss = self.training_step(data)
                epoch_loss.append(loss)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(epoch_loss)
            wandb.log({"loss": epoch_loss, "validation_loss": self.validate(validation_data),
                       "anormal_loss": self.validate(a_normal)})
        return losses

    def training_step(self, data):
        inputs, targets = data
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        scores = torch.sum((outputs - targets) ** 2,  dim=tuple(range(1, outputs.dim())))  # elf.loss_function(outputs, targets)* self.loss_function(outputs, targets)
        loss = torch.mean(scores)
        loss.backward()
        # nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item() / len(inputs)

    def validate(self, validation_data):
        losses = []
        with torch.no_grad():
            for data in validation_data:
                result = self.model(data)
                losses.append(self.loss_function(data, result) * self.loss_function(data, result))
            return sum(losses) / len(validation_data)

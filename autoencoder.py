import torch
from torch import nn
from utils import norm
from torch.optim import lr_scheduler
import wandb


class AutoEncoder(nn.Module):
    def __init__(self, layers, device):
        super().__init__()
        self.layers = layers.to(device)

    def forward(self, x):
        return self.layers(x)


class AETrainer:
    def __init__(self, model, device, epochs, learning_rate):
        self.model = model
        self.device = device
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.loss_function = nn.L1Loss(reduction='sum').to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-6)
        self.lr_scheduler = lr_scheduler.StepLR(self.optimizer, step_size=30000, gamma=0.8)

    def train(self, training_data, validation_data):
        train_loader = torch.utils.data.DataLoader(list(zip(training_data, training_data)), batch_size=1, shuffle=True)
        losses = []
        for _ in range(0, self.epochs):
            epoch_loss = []
            for i, data in enumerate(train_loader, 0):
                loss = self.training_step(data)
                epoch_loss.append(loss)
            epoch_loss = sum(epoch_loss) / len(epoch_loss)
            losses.append(epoch_loss)
            wandb.log({"loss": epoch_loss, "validation_loss": self.validate(validation_data)})
        return losses

    def training_step(self, data):
        inputs, targets = data
        self.optimizer.zero_grad()
        outputs = self.model(norm(inputs))
        targets = norm(targets)
        loss = self.loss_function(outputs, targets) * self.loss_function(outputs, targets)
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()
        return loss.item()

    def validate(self, validation_data):
        losses = []
        with torch.no_grad():
            for data in validation_data:
                t_data = torch.nn.functional.normalize(data, p=2, dim=0)
                result = self.model(t_data)
                losses.append(self.loss_function(t_data, result) * self.loss_function(t_data, result))
            return sum(losses) / len(validation_data)

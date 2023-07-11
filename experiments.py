import csv
import os

import ripserplusplus as rpp_py
import numpy as np
import pickle

import torch

import utils
import wandb
from autoencoder import AutoEncoder, AETrainer
from deepsvdd import DeepSVDD, DeepSVDDTrainer
from utils import norm
from math import sqrt

from torch import nn
import random

device = 'cuda'


def validate_model(trainer, validation_data):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    points = []

    for (data, label) in validation_data:
        latent_point = trainer.model.forward(data)
        if label == "benign":
            points.append(data)
        normal = in_sphere(trainer.c, trainer.R, latent_point)
        if normal and label == "benign":
            true_negative += 1
        if normal and label == 'malware':
            false_negative += 1
        if not normal and label == "benign":
            false_positive += 1
        if not normal and label == 'malware':
            true_positive += 1

    # utils.plot(points, trainer, 250)
    print("true positive", true_positive)
    print("false positive", false_positive)
    print("false negative", false_negative)
    print("true negative", true_negative)
    print(len(validation_data))

    print("accuracy", (true_positive + true_negative) / len(validation_data))
    print("f1-score", (2 * true_positive) / (2 * true_positive + false_positive + false_negative))

    wandb.run.summary["true positive"] = true_positive
    wandb.run.summary["false positive"] = false_positive
    wandb.run.summary["false negative"] = false_negative
    wandb.run.summary["true negative"] = true_negative

    wandb.run.summary["accuracy"] = (true_positive + true_negative) / len(validation_data)
    wandb.run.summary["f1-score"] = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)


def in_sphere(c, r, vec):
    return np.sum((c.cpu().detach().numpy() - vec.cpu().detach().numpy()) ** 2) < r ** 2


def get_trainings_data(values):
    dimension_two = values[2]
    dimension_one = values[1]
    dimension_zero = values[0]

    feature_vec = []

    for x in dimension_two:
        # print(x[1] - x[0], "dim two")
        feature_vec.append(x[0])
        feature_vec.append(x[1])

    for x in dimension_one:
        # print(x[1] - x[0], "dim one")
        feature_vec.append(x[0])
        feature_vec.append(x[1])

    i = -1
    while len(feature_vec) < 20 and -i <= len(dimension_zero):
        # print(dimension_zero[i][1] - dimension_zero[i][0], "dim zero")
        feature_vec.append(dimension_zero[i][0])
        feature_vec.append(dimension_zero[i][1])
        i -= 1

    return feature_vec


def main(latent_space_size, nu, goal, epochs_ae, epoch_dsvdd, lr_ae, lr_dsvdd, batchsize_ae, batchsize_dsvdd,
         normalize):
    with open('precomputed_labeled', 'rb') as f:
        labeled = pickle.load(f)

    trainings_data = []
    validation_data = []

    for (data, label) in labeled:
        data = get_trainings_data(data)
        if len(data) == 20:
            validation_data.append((torch.tensor(data, dtype=torch.float32).to(device), label))

    autoencoder = AutoEncoder(nn.Sequential(
        nn.Linear(20, 15, bias=False),
        nn.LeakyReLU(),
        nn.Linear(15, 10, bias=False),
        nn.LeakyReLU(),
        nn.Linear(10, latent_space_size, bias=False),
        nn.LeakyReLU(),
        nn.Linear(latent_space_size, 10, bias=False),
        nn.LeakyReLU(),
        nn.Linear(10, 15, bias=False),
        nn.LeakyReLU(),
        nn.Linear(15, 20, bias=False),
    ), device)

    learning_rate_ae = lr_ae
    learning_rate_dsvdd = lr_dsvdd
    epochs_autoencoder = epochs_ae
    epochs_dsvdd = epoch_dsvdd
    trainer = AETrainer(autoencoder, device, epochs_autoencoder, learning_rate_ae, batchsize_ae)

    wandb.login(key="")
    run = wandb.init(project="anomaly-topology", config={
        "learning_rate_ae": learning_rate_ae,
        "architecture": "DNN",
        "dataset": "scenario-one-only-benign",
        "goal": goal,
        "epochs_ae": epochs_autoencoder,
        "epochs_dsvdd": epochs_dsvdd,
        "layers_count": len(autoencoder.layers),
        "latent_space_size": latent_space_size,
        "nu": nu
    })

    t_data = list(filter(lambda x: x[1] == 'benign', validation_data))
    t_data = list(map(lambda x: x[0], t_data))

    if normalize:
        utils.norm(t_data)

    wandb.watch(autoencoder)
    random.shuffle(t_data)
    print(len(t_data))
    training = t_data[:400]
    validation = t_data[400:]

    trainer.train(training, validation)

    dsvdd = DeepSVDD(nn.Sequential(nn.Linear(20, 15, bias=False),
                                   nn.LeakyReLU(),
                                   nn.Linear(15, 10, bias=False),
                                   nn.LeakyReLU(),
                                   nn.Linear(10, latent_space_size, bias=False)),
                     autoencoder,
                     device)

    dsvdd_trainer = DeepSVDDTrainer(dsvdd, goal, latent_space_size, learning_rate_dsvdd, nu, epochs_dsvdd,
                                    batchsize_dsvdd,
                                    device)
    dsvdd_trainer.set_center(training)
    dsvdd_trainer.train(training)
    dists = dsvdd_trainer.model.forward(torch.stack(training)) - dsvdd_trainer.c
    dists = dists ** 2
    dsvdd_trainer.R = torch.max(torch.sum(dists, dim=1).sqrt())
    validate_model(dsvdd_trainer, validation_data)
    torch.save(dsvdd_trainer.model.state_dict(), './dsvdd-model')
    torch.save(trainer.model.state_dict(), './ae-model')

    artifact = wandb.Artifact('ae-model', type='model')
    artifact.add_file('ae-model')
    run.log_artifact(artifact)
    artifact = wandb.Artifact('dsvdd-model', type='model')
    artifact.add_file('dsvdd-model')
    run.log_artifact(artifact)

    wandb.finish()


if __name__ == '__main__':
    j = 0.01
    for i in range(2, 8):
        while j <= 0.8:
            main(i, j, 'one-class', 150, 250, 0.001, 0.0001, 50, 75, False)
            j += 0.05

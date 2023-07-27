import csv
import os
import shutil
import sys

import numpy
import ripserplusplus as rpp_py
import numpy as np
import pickle

import torch

import utils
import wandb
from autoencoder import AutoEncoder, AETrainer
from deepsvdd import DeepSVDD, DeepSVDDTrainer
from sklearn import preprocessing

from torch import nn
import random

device = 'cuda'


def validate_model(trainer, validation_data, normalize):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    points_normal = []
    points_anormal = []

    labels = []
    average_anormal_count = []
    fn_labels = set()

    for (data, label, count) in validation_data:
        if normalize:
            latent_point = trainer.model.forward(torch.nn.functional.normalize(data, p=2, dim=0))
        else:
            latent_point = trainer.model.forward(data)
        if label == "benign":
            points_normal.append(data)
            labels.append(label)
        else:
            points_anormal.append(data)
            labels.append(label)

        normal = in_sphere(trainer.c, trainer.R, latent_point)
        if normal and label == "benign":
            true_negative += 1
        if normal and label != 'benign':
            false_negative += 1
            average_anormal_count.append(count)
            fn_labels.add(label)
        if not normal and label == "benign":
            false_positive += 1
        if not normal and label != 'benign':
            true_positive += 1

    utils.plot_validation(points_normal, points_anormal, trainer, labels)
    average_anormal_count.sort()
    print("true positive", true_positive)
    print("false positive", false_positive)
    print("false negative", false_negative)
    print("true negative", true_negative)
    print("average anormal count in false negative", sum(average_anormal_count) / len(average_anormal_count))
    print("median anormal count in false negative", average_anormal_count[len(average_anormal_count) // 2])
    print("false-negative labels", fn_labels)

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


def get_trainings_data(values, input_size):
    dimension_two = values[2]
    dimension_one = values[1]
    dimension_zero = values[0]

    feature_vec = []

    for x in dimension_two:
        # print(x[1] - x[0], "dim two")
        if x[1] - x[0] > 500:
            feature_vec.append(x[0])
            feature_vec.append(x[1])

    for x in dimension_one:
        # print(x[1] - x[0], "dim one")
        if x[1] - x[0] > 500:
            feature_vec.append(x[0])
            feature_vec.append(x[1])

    if len(feature_vec) > input_size:
        return feature_vec[:input_size]

    i = -1
    while len(feature_vec) < input_size and -i <= len(dimension_zero):
        # print(dimension_zero[i][1] - dimension_zero[i][0], "dim zero")
        if dimension_zero[i][1] - dimension_zero[i][0] > 500:
            feature_vec.append(dimension_zero[i][0])
            feature_vec.append(dimension_zero[i][1])
            i -= 1

    return feature_vec


def print_values(training, a_normal, scaler, trainer):
    for i in range(0, 5):
        trainer.model.eval()
        print("normal:", scaler.inverse_transform(training[i].cpu().detach().numpy().reshape(1, -1)))
        # print(scaler.inverse_transform(a_normal[0].cpu().detach().numpy().reshape(1, -1)))
        print("normale ae",
              scaler.inverse_transform(trainer.model.forward(training[i]).cpu().detach().numpy().reshape(1, -1)))
    # print(scaler.inverse_transform(
    #    trainer.model.forward(torch.stack(a_normal)).cpu().detach().numpy()[0].reshape(1, -1)))


def main(latent_space_size, nu, goal, epochs_ae, epoch_dsvdd, lr_ae, lr_dsvdd, batchsize_ae, batchsize_dsvdd,
         normalize, dataset, input_size):
    with open(dataset, 'rb') as f:
        labeled = pickle.load(f)

    validation_data = []
    scaler_data = []

    for (data, label, count) in labeled:
        data = get_trainings_data(data, input_size)
        if len(data) == input_size:
            scaler_data.append(data)
            validation_data.append((torch.tensor(data, dtype=torch.float32).to(device), label, count))

    print(len(validation_data))
    scaler = preprocessing.StandardScaler().fit(scaler_data)
    scaler_data = scaler.transform(scaler_data)

    new_validation_data = []
    labels = set()
    for data, (_, label, count) in zip(scaler_data, validation_data):
        labels.add(label)
        new_validation_data.append((torch.tensor(data, dtype=torch.float32).to(device), label, count))

    validation_data = new_validation_data

    autoencoder = AutoEncoder(nn.Sequential(
        nn.Linear(input_size, 10, bias=False),
        nn.LeakyReLU(),
        nn.Linear(10, latent_space_size, bias=False),
        nn.LeakyReLU(),
        nn.Linear(latent_space_size, 10, bias=False),
        nn.Dropout(0.2),
        nn.LeakyReLU(),
        nn.Linear(10, input_size, bias=False)
    ), device)

    shutil.rmtree("state")
    os.mkdir("state")

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
    a_normal_data = list(filter(lambda x: x[1] != 'benign', validation_data))
    a_normal_data = list(map(lambda x: x[0], a_normal_data))

    wandb.watch(autoencoder)
    random.shuffle(t_data)
    training = t_data[:int(len(t_data) * 0.9)]
    validation = t_data[int(len(t_data) * 0.9):]

    trainer.train(training, validation, a_normal_data)
    print_values(training, a_normal_data, scaler, trainer)

    dsvdd = DeepSVDD(nn.Sequential(nn.Dropout(),
                                   nn.Linear(input_size, 10, bias=False),
                                   nn.LeakyReLU(),
                                   nn.Linear(10, latent_space_size, bias=False)),
                     autoencoder,
                     device)

    dsvdd_trainer = DeepSVDDTrainer(dsvdd, goal, latent_space_size, learning_rate_dsvdd, nu, epochs_dsvdd,
                                    batchsize_dsvdd,
                                    device)
    dsvdd_trainer.set_center(training)
    dsvdd_trainer.train(training, validation_data)
    if not normalize:
        training = torch.stack(training)

    if goal == 'one-class':
        dists = dsvdd_trainer.model.forward(training) - dsvdd_trainer.c
        dists = dists ** 2
        # Let's take a quantile of our trainings data as radius
        dsvdd_trainer.R = torch.quantile(torch.sum(dists, dim=1).sqrt(), 0.9)
        print(dsvdd_trainer.R)
    validate_model(dsvdd_trainer, validation_data, normalize)
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
    dataset = sys.argv[1]

    # for i in range(100, 1000, 50):
    # torch.manual_seed(42)
    main(8, 0.01, 'one-class', 1000, 500, 1e-4, 1e-4, 15, 30, False, dataset, 10)

    # j = 0.01
    # for i in range(2, 8):
    #    while j <= 0.8:
    #        main(i, j, 'soft-boundary', 150, 250, 0.001, 0.0001, 50, 75, False, dataset)
    #        main(i, j, 'soft-boundary', 150, 250, 0.001, 0.0001, 50, 75, True, dataset)
    #        j += 0.05
#
# for i in range(2, 8):
#    main(i, 0.1, 'one-class', 150, 250, 0.001, 0.0001, 50, 75, False, dataset)
#    main(i, 0.1, 'one-class', 150, 250, 0.001, 0.0001, 50, 75, True, dataset)

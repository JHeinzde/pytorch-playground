import csv
import ripserplusplus as rpp_py
import numpy as np
import pickle

import torch
import wandb
from autoencoder import AutoEncoder, AETrainer
from deepsvdd import DeepSVDD, DeepSVDDTrainer
from utils import norm
from math import sqrt

from torch import nn
import random

device = 'cuda'


def validate_model(dsvdd, c, r, validation_data):
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for (data, label) in validation_data:
        latent_point = dsvdd.forward(torch.nn.functional.normalize(data, p=2, dim=0))
        normal = in_sphere(c, r, latent_point)
        if normal and label == "benign":
            true_negative += 1
        if normal and label == 'malware':
            false_negative += 1
        if not normal and label == "benign":
            false_positive += 1
        if not normal and label == 'malware':
            true_positive += 1

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


def prepare(scenario, window_size):
    result = []
    for i in range(0, len(scenario)):
        result.append(rpp_py.run("--format point-cloud --dim 5", scenario[i:i + window_size]))
    return result


def prepare_labeled(scenario, windows_size):
    result = []
    for i in range(0, len(scenario)):
        label = 'benign'
        tmp = scenario[i:i + windows_size]
        new = []
        for x in tmp:
            new.append(x[:-1])
        new = np.array(list(map(lambda x: list(map(lambda y: float(y), x)), new)))
        for t in tmp:
            if t[-1] != 'benign':
                label = 'malware'
        res = rpp_py.run("--format point-cloud --dim 5", new)
        result.append((res, label))
    return result


def get_trainings_data(values):
    dimension_two = values[2]
    dimension_one = values[1]
    dimension_zero = values[0]

    feature_vec = []

    for x in dimension_two:
        feature_vec.append(x[0])
        feature_vec.append(x[1])

    for x in dimension_one:
        feature_vec.append(x[0])
        feature_vec.append(x[1])

    i = -1
    while len(feature_vec) < 20 and -i <= len(dimension_zero):
        feature_vec.append(dimension_zero[i][0])
        feature_vec.append(dimension_zero[i][1])
        i -= 1

    return feature_vec


def main():
    scenario_one = []
    scenario_one_all_data = []
    scenario_two = []
    with open('scenario1.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[-1] == "benign":
                scenario_one.append(row[:-1])
            scenario_one_all_data.append(row)

    with open('scenario2.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            scenario_two.append(row)

    scenario_one = scenario_one[1:]
    scenario_one = np.array(list(map(lambda x: list(map(lambda y: float(y), x)), scenario_one)))
    # res = prepare(scenario_one, 30)
    # labeled = prepare_labeled(scenario_one_all_data[1:], 30)
    # print(labeled)
    # with open('precomputed_labeled', 'wb+') as f:
    #   pickle.dump(labeled, f)
    # with open('precomputed', 'wb+') as f:
    #    pickle.dump(res, f)

    with open('precomputed', 'rb') as f:
        res = pickle.load(f)
    with open('precomputed_labeled', 'rb') as f:
        labeled = pickle.load(f)

    trainings_data = []
    validation_data = []

    latent_space_size = 3



    for r in res:
        data = get_trainings_data(r)
        if len(data) == 20:
            trainings_data.append(torch.tensor(data, dtype=torch.float32).to(device))

    for (data, label) in labeled:
        data = get_trainings_data(data)
        if len(data) == 20:
            validation_data.append((torch.tensor(data, dtype=torch.float32).to(device), label))

    list(filter(lambda x: ))
    return

    autoencoder = AutoEncoder(nn.Sequential(
        nn.Linear(20, 10, bias=False),
        nn.Sigmoid(),
        nn.Linear(10, 9, bias=False),
        nn.Sigmoid(),
        nn.Linear(9, 8, bias=False),
        nn.Sigmoid(),
        nn.Linear(8, latent_space_size, bias=False),
        nn.Sigmoid(),
        nn.Linear(latent_space_size, 8, bias=False),
        nn.Sigmoid(),
        nn.Linear(8, 9, bias=False),
        nn.Sigmoid(),
        nn.Linear(9, 10, bias=False),
        nn.Sigmoid(),
        nn.Linear(10, 20, bias=False),
    ), device)

    learning_rate = 0.01
    epochs = 150
    trainer = AETrainer(autoencoder, device, epochs, learning_rate)

    wandb.login(key="6a208e4a933bae5c70308d10d911239d16d52a13")
    wandb.init(project="anomaly-topology", config={
        "learning_rate": learning_rate,
        "architecture": "DNN",
        "dataset": "scenario-one-only-benign",
        "epochs": epochs,
        "layers_count": len(autoencoder.layers),
        "latent_space_size": latent_space_size
    })

    t_data = list(filter(lambda x: x[1] == 'benign', validation_data))

    wandb.watch(autoencoder)
    random.shuffle(t_data)
    training = t_data[:400]  # trainings_data[:704]
    validation = t_data[400:]

    training = list(map(lambda x: x[0], training))
    validation = list(map(lambda x: x[0], validation))

    trainer.train(training, validation)

    dsvdd = DeepSVDD(nn.Sequential(nn.Linear(20, 10, bias=False),
                                   nn.Sigmoid(),
                                   nn.Linear(10, 9, bias=False),
                                   nn.Sigmoid(),
                                   nn.Linear(9, 8, bias=False),
                                   nn.Sigmoid(),
                                   nn.Linear(8, latent_space_size, bias=False)),
                     autoencoder,
                     device)

    dsvdd_trainer = DeepSVDDTrainer(dsvdd, 'soft-boundary', latent_space_size, learning_rate, 50, device)
    dsvdd_trainer.set_center(training)
    dsvdd_trainer.train(training)
    validate_model(dsvdd, dsvdd_trainer.c, dsvdd_trainer.R, validation_data)

    wandb.finish()


if __name__ == '__main__':
    # c = np.array([1, 2])
    # R = 2
    #
    # p = np.array([1, 2.5])
    #
    # in_sphere = np.sum((c - p) ** 2) < R ** 2
    #
    # print(in_sphere)

    main()

import csv
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy
import ripserplusplus as rpp_py
import numpy as np
import pickle
import json

import torch

import utils
import wandb
from autoencoder import AutoEncoder, AETrainer
from deepsvdd import DeepSVDD, DeepSVDDTrainer
from sklearn import preprocessing

from torch import nn
import random

from hsc import HSCNet, HSCTrainer

device = 'cuda'


def validate_model(trainer, validation_data, scaler, autoencoder):
    """
    Validates the model after training
    :param trainer: Deep SVDD trainer with all related params of the model
    :param validation_data: Data for validation
    :param scaler: Scaler to rescale data for plots
    :param autoencoder: The original autoencoder used for visualizing data
    :return: All parameters of the confusion matrix as well as accuracy f1 and mcc scores.
    """
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    points_normal = []
    points_anormal = []

    labels = []
    average_anormal_count = []
    fn_labels = set()

    false_negatives = []
    true_normals = []

    for (data, label, count) in validation_data:
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
            true_normals.append(data)
        if normal and label != 'benign':
            false_negative += 1
            average_anormal_count.append(count)
            fn_labels.add(label)
            false_negatives.append(data)
        if not normal and label == "benign":
            false_positive += 1
        if not normal and label != 'benign':
            true_positive += 1

    # Now we plot the autoencoder presentations of the true negative values vs the false negative values.
    points_x_normal = []
    points_y_normal = []
    points_x_anormal = []
    points_y_anormal = []

    points_x_ae_normal = []
    points_y_ae_normal = []
    points_x_ae_anormal = []
    points_y_ae_anormal = []

    for d in false_negatives:
        rep = autoencoder.forward(d)
        d = d.cpu().detach().numpy()
        rep = rep.cpu().detach().numpy()
        d = scaler.inverse_transform([d])[0]
        rep = scaler.inverse_transform([rep])[0]
        for i in range(0, len(d), 2):
            points_x_anormal.append(d[i])
            points_y_anormal.append(d[i + 1])

        for i in range(0, len(rep), 2):
            points_x_ae_anormal.append(rep[i])
            points_y_ae_anormal.append(rep[i + 1])

    for d in true_normals:
        rep = autoencoder.forward(d)
        d = d.cpu().detach().numpy()
        rep = rep.cpu().detach().numpy()
        d = scaler.inverse_transform([d])[0]
        rep = scaler.inverse_transform([rep])[0]

        for i in range(0, len(d), 2):
            points_x_normal.append(d[i])
            points_y_normal.append(d[i + 1])

        for i in range(0, len(rep), 2):
            points_x_ae_normal.append(rep[i])
            points_y_ae_normal.append(rep[i + 1])

    figure, axes = plt.subplots()

    plt.scatter(points_x_ae_anormal, points_y_ae_anormal, color="red", s=2, label="autoencoder output anomalous data")
    plt.scatter(points_x_ae_normal, points_y_ae_normal, color="green", s=1, label="autoencoder output normal data")
    axes.set_xlabel("Birth")
    axes.set_ylabel("Death")
    axes.legend()
    plt.savefig(f'false-negatives', bbox_inches='tight')
    plt.close(figure)
    figure, axes = plt.subplots()
    plt.scatter(points_x_anormal, points_y_anormal, color="red", s=1, label="original anomalous data")
    plt.scatter(points_x_normal, points_y_normal, color="green", s=2, label="original normal data")
    axes.set_xlabel("Birth")
    axes.set_ylabel("Death")
    axes.legend()
    plt.savefig(f'original-data-false-negatives', bbox_inches='tight')
    plt.close(figure)

    average_anormal_count.sort()

    print("true positive", true_positive)
    print("false positive", false_positive)
    print("false negative", false_negative)
    print("true negative", true_negative)
    print("average anormal count in false negative", sum(average_anormal_count) / len(average_anormal_count))
    print("median anormal count in false negative", average_anormal_count[len(average_anormal_count) // 2])
    print("false-negative labels", fn_labels)

    acc = (true_positive + true_negative) / len(validation_data)
    print("accuracy", acc)
    f_one = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
    print("f1-score", f_one)
    mcc = (true_positive * true_negative - false_positive * false_negative) / np.sqrt(
        (true_positive + false_positive) * (true_positive + false_negative) * (true_negative + false_positive) * (
                true_negative + false_positive))
    print("mcc", mcc)

    wandb.run.summary["true positive"] = true_positive
    wandb.run.summary["false positive"] = false_positive
    wandb.run.summary["false negative"] = false_negative
    wandb.run.summary["true negative"] = true_negative
    wandb.run.summary["accuracy"] = (true_positive + true_negative) / len(validation_data)
    wandb.run.summary["f1-score"] = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
    wandb.run.summary["MCC"] = mcc

    return true_positive, false_positive, false_negative, true_negative, acc, f_one, mcc


def in_sphere(c, r, vec):
    """
    Check if a point is inside the hpyerpshere
    :param c: Center of the hypersphere
    :param r:  Radius of the hypersphere
    :param vec: Point to check
    :return: True if the point is inside of the sphere false otherwise
    """
    return np.sum((c.cpu().detach().numpy() - vec.cpu().detach().numpy()) ** 2) < r ** 2


def get_trainings_data(values, input_size, cut_off):
    """
    Build training vectors
    :param values: Original persistence diagram
    :param input_size: Desired size of training vectors
    :param cut_off: Every lifetime smaller than this value will be filtered out
    :return: The desired trainings vector
    """
    dimension_two = values[2]
    dimension_one = values[1]
    dimension_zero = values[0]

    feature_vec = []

    avg_life_time = 0
    count = 0

    for x in dimension_two:
        avg_life_time += x[1] - x[0]
        count += 1

    for x in dimension_one:
        avg_life_time += x[1] - x[0]
        count += 1

    for x in dimension_zero:
        avg_life_time += x[1] - x[0]
        count += 1

    for x in dimension_two:
        if x[1] - x[0] > cut_off:
            # print(x[1] - x[0], "dim two"
            feature_vec.append(x[0])
            feature_vec.append(x[1])

    for x in dimension_one:
        if x[1] - x[0] > cut_off:
            # print(x[1] - x[0], "dim one")
            feature_vec.append(x[0])
            feature_vec.append(x[1])

    if len(feature_vec) > input_size:
        return feature_vec[:input_size]

    i = -1
    cutoff = 0
    while len(feature_vec) < input_size and -i < len(dimension_zero):
        if dimension_zero[i][1] - dimension_zero[i][0] > cut_off:
            feature_vec.append(dimension_zero[i][0])
            feature_vec.append(dimension_zero[i][1])
            i -= 1
        else:
            cutoff += 1

    return feature_vec


def main(latent_space_size, nu, goal, epochs_ae, epoch_dsvdd, lr_ae, lr_dsvdd, batchsize_ae, batchsize_dsvdd,
          dataset, input_size):
    """
    Main method for experiment. Loads a data set builds training data, splits training and validation data.
    Then trains an auotencoder and then a deep svdd model
    :param latent_space_size: Size of the latent space for the autoencoder and the Deep SVD model
    :param nu: The nu value used for this experiment run
    :param goal: The training goal for this run
    :param epochs_ae: The amount of epochs for the autoencoder
    :param epoch_dsvdd: The amount fo epochs for Deep SVDD
    :param lr_ae: Initial learning rate for the autoencoder
    :param lr_dsvdd: Initial learning rate for the Deep SVDD model
    :param batchsize_ae: Batchsize for the autoencoder
    :param batchsize_dsvdd: Batchsize for the Deep SVDD training
    :param dataset: The data set that should be loaded
    :param input_size: The size of the training vectors
    :return: All parameters of the validation
    """
    with open(dataset, 'rb') as f:
        labeled = pickle.load(f)

    validation_data_normal = []
    validation_data_anormal = []
    scaler_data = []
    anormal_scaler_data = []

    # Separate data by class
    for (data, label, count) in labeled:
        data = get_trainings_data(data, input_size, 500)
        if len(data) == input_size:
            if label == 'benign':
                scaler_data.append(data)
                validation_data_normal.append((torch.tensor(data, dtype=torch.float32).to(device), label, count))
            else:
                anormal_scaler_data.append(data)
                validation_data_anormal.append((torch.tensor(data, dtype=torch.float32).to(device), label, count))

    # scale data
    scaler = preprocessing.StandardScaler().fit(scaler_data)
    scaler_data = scaler.transform(scaler_data)
    anormal_scaler_data = scaler.transform(anormal_scaler_data)

    new_validation_data = []
    labels = set()
    for data, (_, label, count) in zip(scaler_data, validation_data_normal):
        labels.add(label)
        new_validation_data.append((torch.tensor(data, dtype=torch.float32).to(device), label, count))
    for data, (_, label, count) in zip(anormal_scaler_data, validation_data_anormal):
        labels.add(label)
        new_validation_data.append((torch.tensor(data, dtype=torch.float32).to(device), label, count))

    validation_data = new_validation_data

    autoencoder = AutoEncoder(nn.Sequential(
        nn.Linear(input_size, 15, bias=False),
        nn.LeakyReLU(),
        nn.Linear(15, 10, bias=False),
        nn.LeakyReLU(),
        nn.Linear(10, latent_space_size, bias=False),
        nn.LeakyReLU(),
        nn.Linear(latent_space_size, 10, bias=False),
        nn.LeakyReLU(),
        nn.Linear(10, 15, bias=False),
        nn.LeakyReLU(),
        nn.Linear(15, input_size, bias=False)
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

    # build test and training data splits
    t_data = list(filter(lambda x: x[1] == 'benign', validation_data))
    t_data = list(map(lambda x: x[0], t_data))
    a_normal_data = list(filter(lambda x: x[1] != 'benign', validation_data))
    a_normal_data = list(map(lambda x: x[0], a_normal_data))

    wandb.watch(autoencoder)
    random.shuffle(t_data)
    training = t_data[:int(len(t_data) * 0.8)]
    validation = t_data[int(len(t_data) * 0.8):]

    final_validation = []
    for x in validation_data:
        if any([(x[0] == c_).all() for c_ in validation]):
            final_validation.append(x)
        elif x[1] != 'benign':
            final_validation.append(x)

    trainer.train(training, validation, a_normal_data)

    dsvdd = DeepSVDD(nn.Sequential(nn.Linear(input_size, 10, bias=False),
                                   nn.LeakyReLU(),
                                   nn.Linear(15, 10, bias=False),
                                   nn.LeakyReLU(),
                                   nn.Linear(10, latent_space_size, bias=False), ),
                     autoencoder,
                     device)

    dsvdd_trainer = DeepSVDDTrainer(dsvdd, goal, latent_space_size, learning_rate_dsvdd, nu, epochs_dsvdd,
                                    batchsize_dsvdd,
                                    device)
    dsvdd_trainer.set_center(training)
    dsvdd_trainer.train(training, final_validation)
    training = torch.stack(training)

    # If one class set final radius to the 1-nu quantile of distances from training data to center
    if goal == 'one-class':
        dists = dsvdd_trainer.model.forward(training) - dsvdd_trainer.c
        dists = dists ** 2
        # Let's take a quantile of our trainings data as radius
        dsvdd_trainer.R = torch.quantile(torch.sum(dists, dim=1).sqrt(), 1 - nu)
    tp, fp, fn, tn, acc, f_one, mcc = validate_model(dsvdd_trainer, final_validation, scaler, autoencoder)
    torch.save(dsvdd_trainer.model.state_dict(), './dsvdd-model')
    torch.save(trainer.model.state_dict(), './ae-model')
    artifact = wandb.Artifact('ae-model', type='model')
    artifact.add_file('ae-model')
    run.log_artifact(artifact)
    artifact = wandb.Artifact('dsvdd-model', type='model')
    artifact.add_file('dsvdd-model')
    run.log_artifact(artifact)
    wandb.finish()
    return tp, fp, fn, tn, acc, f_one, mcc


if __name__ == '__main__':
    dataset = sys.argv[1]
    goal = 'one-class'
    epochs_ae = 500
    epochs_dsvdd = 1000
    tp, fp, fn, tn, acc, f_one, mcc = main(8, 0.01, 'one-class', epochs_ae, epochs_dsvdd, 1e-3, 1e-4, 50, 30, dataset,
                                               50)
import ripserplusplus as rpp_py
import numpy as np
import csv
import torch
import pickle


def prepare(scenario, window_size):
    result = []
    for i in range(0, len(scenario)):
        result.append(rpp_py.run("--format point-cloud --dim 3", scenario[i:i + window_size]))
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
        res = rpp_py.run("--format point-cloud --dim 3", new)
        result.append((res, label))
    return result


def pre_calc():
    scenario_one = []
    scenario_one_all_data = []
    scenario_validation = []
    scenario_two = []
    with open('scenario1.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[-1] == "benign":
                scenario_one.append(row[:-1])
                scenario_validation.append((row[:-1], 'benign'))
            else:
                scenario_validation.append((row[:-1], 'malware'))
            scenario_one_all_data.append(row)

    with open('scenario2.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            scenario_two.append(row)

    scenario_one = scenario_one[1:]
    scenario_validation = scenario_validation[1:]
    scenario_validation = list(
        map(lambda x: (torch.tensor(list(map(lambda y: float(y), x[0])), dtype=torch.float32).to(device), x[1]),
            scenario_validation))

    scenario_one = list(map(lambda x: list(map(lambda y: float(y), x)), scenario_one))
    # res = prepare(scenario_one, 50)
    labeled = prepare_labeled(scenario_one_all_data[1:], 10)
    with open('precomputed_labeled-10', 'wb+') as f:
        pickle.dump(labeled, f)
    # with open('precomputed-50', 'wb+') as f:
#   pickle.dump(res, f)

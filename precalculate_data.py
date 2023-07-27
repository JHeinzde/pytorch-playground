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
        a_normal_count = 0
        for x in tmp:
            new.append(x[:-1])
        new = np.array(list(map(lambda x: list(map(lambda y: float(y), x)), new)))
        for t in tmp:
            if t[-1] != 'benign':
                label = t[-1]
                a_normal_count += 1
        res = rpp_py.run("--format point-cloud --dim 3", new)
        result.append((res, label, a_normal_count))
    return result


def pre_calc(window_size):
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
                scenario_validation.append((row[:-1], row[-1]))
            scenario_one_all_data.append(row)

    with open('scenario2.csv') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            scenario_two.append(row)

    labeled = prepare_labeled(scenario_one_all_data[1:], window_size)
    with open(f'precomputed_labeled-{window_size}', 'wb+') as f:
        pickle.dump(labeled, f)


def main():
    for window_size in range(10, 60, 10):
        pre_calc(window_size)


if __name__ == '__main__':
    main()

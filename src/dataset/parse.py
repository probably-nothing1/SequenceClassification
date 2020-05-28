import numpy as np

def parse_data(filename):
    xs = []
    ys = []
    with open(filename) as f:
        for line in f.readlines():
            line = line[:-1].replace(',', '')
            line = [int(x) for x in line.replace('-', '')]
            x = line[::2]
            y = line[1::2]
            xs.append(x)
            ys.append(y)
    return xs, ys

def parse_and_preprocess_data(filename):
    """
    Parse data and map each point form the lattice to [0..99]
    using fixed permutation
    """
    np.random.seed(1337)
    permutation = np.random.permutation(100)
    data = []
    with open(filename) as f:
        for line in f.readlines():
            line = line[:-1].replace(',', '')
            line = [int(x) for x in line.replace('-', '')]
            xs = line[::2]
            ys = line[1::2]
            indices = [x + y * 10 for x, y in zip(xs,ys)]
            data.append([permutation[index] for index in indices])
    return data

def parse_labels(filename):
    labels = []
    with open(filename) as f:
        for line in f.readlines():
            labels.append(int(line[:-1]))

    return labels
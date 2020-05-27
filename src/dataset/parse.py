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

def parse_labels(filename):
    labels = []
    with open(filename) as f:
        for line in f.readlines():
            labels.append(int(line[:-1]))

    return labels
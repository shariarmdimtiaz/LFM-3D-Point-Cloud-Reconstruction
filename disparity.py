import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error


def output_disparity(output, label):
    label = label[0]
    pad1_half = int(0.5 * (np.size(label, 1) - np.size(output, 1)))
    label482 = label[:, 15:-15, 15:-15]

    output_data = output[ 15 - pad1_half:482 + 15 - pad1_half, 15 - pad1_half:482 + 15 - pad1_half]
    output482 = [output_data]

    diff = np.abs(output482 - label482)
    bp7 = (diff >= 0.07)
    bp3 = (diff >= 0.03)
    bp1 = (diff >= 0.01)

    mse = mean_squared_error(label482[0], output482[0])
    return mse
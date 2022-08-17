import numpy as np


def calculate_custom_metric(l: list) -> float:
    max = np.max(l)
    min = np.min(l)
    mean = np.mean(l)

    print("Custom: " + str(mean - (max - min)))

    return mean - (max - min)


def test_metric_1():
    l = [1000, 900, 800, 400, 300, 110, -80, -270, -460]

    assert calculate_custom_metric(l) == -1160


def test_metric_2():
    l = [-1000, -900, -800, -400, -300, -110, -80, -270, -460]

    assert calculate_custom_metric(l) == -1400


def test_metric_3():
    l = [1000, 950, 950, 950, 900, 900]

    assert calculate_custom_metric(l) == 841.6666666666666

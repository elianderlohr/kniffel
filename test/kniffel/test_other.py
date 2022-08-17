import numpy as np


def calculate_custom_metric(l: list) -> float:
    max = np.max(l)
    min = np.min(l)
    mean = np.mean(l)

    print("Custom: " + str((((max - min) * mean) / 1_000) - abs(mean)))

    return (((max - min) * mean) / 1_000) - abs(mean)


def test_metric_1():
    l = [1000, 900, 800, 400, 300, 110, -80, -270, -460]

    assert calculate_custom_metric(l) == 138


def test_metric_2():
    l = [-1000, -900, -800, -400, -300, -110, -80, -270, -460]

    assert calculate_custom_metric(l) == (((-80 - -1000) * -480) / 1_000) - abs(-480)

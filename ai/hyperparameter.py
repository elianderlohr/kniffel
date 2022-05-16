import itertools
import random
import numpy as np


class Hyperparameter:
    _randomize = False
    _predefined_layers = False

    hps = list()

    units = list(range(16, 64, 16))

    base_hp = {
        "windows_length": [13],
        "adam_learning_rate": np.arange(0.0001, 0.001, 0.0002),
        "batch_size": [32],
        "target_model_update": np.arange(0.0001, 0.001, 0.0002),
        "dueling_option": ["avg"],
        "activation": ["linear"],
        "layers": [1],
        "unit_1": units,
        "unit_2": [1],
    }

    def __init__(self, predefined_layers=False, randomize=False) -> None:
        self._predefined_layers = predefined_layers
        self._randomize = randomize
        self._create_product()

    def _pepare(self):
        for i in range(max(list(self.base_hp["layers"]))):
            self.base_hp["unit_" + str(i)] = self.units

    def _create_product(self):
        if self._predefined_layers is False:
            self._pepare()

        values = list(itertools.product(*self.base_hp.values()))

        values_mod = [{i: j for i, j in zip(self.base_hp.keys(), v)} for v in values]

        self.hps = [d for d in values_mod if d["unit_1"] >= d["unit_2"]]

        if self._randomize:
            self.hps = random.sample(self.hps, k=len(self.hps))

        print(f"Created {len(self.hps)} combinations to test.")

        return self.hps

    def add_hp(self, key, value):
        self.base_hp[key] = value

    def get(self):
        return self._create_product()

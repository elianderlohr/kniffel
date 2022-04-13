import itertools
import random


class Hyperparameter:
    _randomize = False

    hps = list()

    base_hp = {
        "windows_length": [1],
        "adam_learning_rate": [0.0001],
        "batch_size": [20],
        "target_model_update": [0.0001],
        "dueling_option": ["avg"],
        "eps": [0.2],
        "activation": ["linear", "softmax"],
        "layers": range(1, 5),
        "units": {
            "1": list(range(16, 128, 16)),
            "2": list(range(16, 128, 16)),
            "3": list(range(16, 128, 16)),
            "4": list(range(16, 128, 16)),
            "5": list(range(16, 128, 16)),
        },
    }

    def __init__(self, randomize=False) -> None:
        self._randomize = randomize
        self._create_product()

    def _create_product(self):
        self.hps = list(itertools.product(self.base_hp.values()))

        if self._randomize:
            self.hps = random.sample(self.hps, k=len(self.hps))

        return self.hps

    def add_hp(self, key, value):
        self.base_hp[key] = value

    def get(self):
        return self._create_product()

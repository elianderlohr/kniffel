import itertools
import random


class Hyperparameter:
    _randomize = False

    hps = list()
    
    units = list(range(16, 128, 16))
    
    base_hp = {
        "windows_length": [1],
        "adam_learning_rate": [0.0001],
        "batch_size": [20],
        "target_model_update": [0.0001],
        "dueling_option": ["avg"],
        "eps": [0.2],
        "activation": ["linear", "softmax"],
        "layers": range(2, 6), 
    }

    def __init__(self, randomize=False) -> None:
        self._randomize = randomize
        self._create_product()
    
    def _pepare(self):
        for i in range(max(list(self.base_hp["layers"]))):
            self.base_hp["unit_" + str(i)] = self.units
    
    def _create_product(self):
        self._pepare()
        
        values = list(itertools.product(*self.base_hp.values()))

        values_mod = [ {i: j for i, j in zip(self.base_hp.keys(), v) } for v in values ]
        
        self.hps = values_mod

        if self._randomize:
            self.hps = random.sample(self.hps, k=len(self.hps))

        return self.hps

    def add_hp(self, key, value):
        self.base_hp[key] = value

    def get(self):
        return self._create_product()
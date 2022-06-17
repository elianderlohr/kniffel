import warnings
import numpy as np
from ai import KniffelAI

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    units = list(range(16, 64, 16))

    base_hp = {
        "windows_length": [1],
        "adam_learning_rate": np.arange(0.0001, 0.001, 0.0002),
        "batch_size": [32],
        "target_model_update": np.arange(0.0001, 0.001, 0.0002),
        "dueling_option": ["avg"],
        "activation": ["linear"],
        "layers": [2],
        "unit_1": units,
        "unit_2": units,
    }

    ai = KniffelAI(
        save=True, load=False, predefined_layers=True, hyperparater_base=base_hp
    )

    env_config = {
        "reward_step": 0,
        "reward_round": 0.5,
        "reward_roll_dice": 0.25,
        "reward_game_over": -10,
        "reward_slash": -5,
        "reward_bonus": 2,
        "reward_finish": 10,
        "reward_zero_dice": -0.5,
        "reward_one_dice": -0.2,
        "reward_two_dice": -0.1,
        "reward_three_dice": 0.5,
        "reward_four_dice": 0.6,
        "reward_five_dice": 0.8,
        "reward_six_dice": 1,
        "reward_kniffel": 1.5,
        "reward_small_street": 1,
        "reward_large_street": 1.1,
    }

    hyperparameter = {
        "windows_length": 1,
        "adam_learning_rate": 0.0005,
        "batch_size": 128,
        "target_model_update": 0.01,
        "adam_epsilon": 0.00001,
        "dueling_option": "avg",
        "activation": "linear",
        "layers": 2,
        "unit_1": 48,
        "unit_2": 48,
    }

    ai.train(hyperparameter=hyperparameter, nb_steps=10_000, env_config=env_config)

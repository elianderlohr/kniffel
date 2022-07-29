# Standard imports
from gc import callbacks
from statistics import mean
from datetime import datetime as dt
from tokenize import Triple
from unittest.mock import call
import warnings
import numpy as np
import json
import optuna
import os
from pathlib import Path
import sys

# Keras imports

import tensorflow as tf

from rl.agents import DQNAgent, CEMAgent, SARSAAgent
from rl.policy import (
    BoltzmannQPolicy,
    EpsGreedyQPolicy,
    LinearAnnealedPolicy,
    GreedyQPolicy,
    MaxBoltzmannQPolicy,
    BoltzmannGumbelQPolicy,
)

from rl.memory import SequentialMemory
from keras.layers import Dense, Flatten
from keras.callbacks import EarlyStopping

from tensorflow.keras.optimizers import Adam

# Kniffel

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.options import KniffelOptions
from src.kniffel.classes.kniffel import Kniffel
from src.ai.hyperparameter import Hyperparameter
from src.ai.env import EnumAction
from src.ai.env import KniffelEnv
import src.kniffel.classes.custom_exceptions as ex


class KniffelAI:
    # Load model from path
    _load = False

    # Hyperparameter object
    _hyperparater_base = {}

    # Test episodes
    _test_episodes = 100

    # path prefix
    _path_prefix = ""

    # Path to config csv
    _config_path = ""

    # Optuna trial
    _trial: optuna.trial.Trial = None
    _trial_tracker: dict = {}
    # Agent
    _agent_value = ""

    # Current date
    datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")

    def __init__(
        self,
        load=False,
        test_episodes=100,
        path_prefix="",
        hyperparater_base={},
        config_path="src/ai/Kniffel.CSV",
        trial=None,
    ):
        self._load = load
        self._hyperparater_base = hyperparater_base
        self._test_episodes = test_episodes
        self._config_path = config_path
        self._trial = trial

        if path_prefix == "":
            try:
                import google.colab

                self._path_prefix = "/"
            except:
                self._path_prefix = ""
        else:
            self._path_prefix = path_prefix

        self._agent_value = self._return_trial("agent")

    def _return_trial(self, key):
        if key not in self._trial_tracker.keys():
            self._trial_tracker[key] = self._trial.suggest_categorical(
                key, self._hyperparater_base[key]
            )

        return self._trial_tracker[key]

    # Model
    def build_model(self, actions):
        model = tf.keras.Sequential()
        model.add(
            Flatten(
                input_shape=(
                    self._return_trial("windows_length"),
                    1,
                    41,
                )
            )
        )

        for i in range(1, self._trial.suggest_int("layers", 1, 4)):
            model.add(
                Dense(
                    self._trial.suggest_int("n_units_l" + str(i), 16, 256),
                    activation="relu",
                )
            )

        model.add(
            Dense(
                actions,
                activation=self._return_trial("activation"),
            )
        )

        model.summary()
        return model

    def get_policy(self, key):

        policy = None
        if key == "LinearAnnealedPolicy":

            policy = LinearAnnealedPolicy(
                EpsGreedyQPolicy(),
                attr="eps",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=1_000_000,
            )

        elif key == "EpsGreedyQPolicy":

            policy = EpsGreedyQPolicy(
                eps=self._trial.suggest_float("greedy_eps", 1e-9, 1e-1)
            )

        elif key == "GreedyQPolicy":

            policy = GreedyQPolicy()

        elif key == "BoltzmannQPolicy":

            policy = BoltzmannQPolicy(
                tau=self._trial.suggest_float("tau", 0.05, 1, step=0.05)
            )

        elif key == "MaxBoltzmannQPolicy":

            policy = MaxBoltzmannQPolicy(
                eps=self._trial.suggest_float("boltzmann_eps", 1e-9, 1e-1),
                tau=self._trial.suggest_float("tau", 0.05, 1, step=0.05),
            )
        elif key == "BoltzmannGumbelQPolicy":

            policy = BoltzmannGumbelQPolicy(
                C=self._trial.suggest_float("C", 0.05, 1, step=0.05)
            )

        return policy

    def build_agent(self, model, actions, nb_steps):
        """Build dqn agent

        :param model: deep neural network model (keras)
        :param actions: action space
        :param nb_steps: steps the model should be trained
        :param hyperparameter: hyperparameter the agent should use
        :return: agent
        """
        agent = None

        memory = SequentialMemory(
            limit=500_000,
            window_length=self._return_trial("windows_length"),
        )

        if self._agent_value == "DQN":
            agent = DQNAgent(
                model=model,
                memory=memory,
                policy=self.get_policy(self._return_trial("train_policy")),
                nb_actions=actions,
                nb_steps_warmup=1_000,
                target_model_update=self._return_trial("target_model_update"),
                batch_size=self._return_trial("batch_size"),
                dueling_type=self._return_trial("dueling_option"),
                enable_double_dqn=self._return_trial("enable_double_dqn"),
            )

        elif self._agent_value == "CEM":
            agent = CEMAgent(
                model=model,
                memory=memory,
                nb_actions=actions,
                nb_steps_warmup=1_000,
                batch_size=self._return_trial("batch_size"),
            )

        elif self._agent_value == "SARSA":
            agent = SARSAAgent(
                model=model,
                policy=self.get_policy(self._return_trial("train_policy")),
                test_policy=self.get_policy(self._return_trial("test_policy")),
                nb_actions=actions,
                nb_steps_warmup=1_000,
                gamma=self._trial.suggest_float("gamma", 0.01, 0.99),
            )

        return agent

    def train_agent(
        self,
        actions,
        env,
        nb_steps,
        load_path="",
    ):
        model = self.build_model(
            actions,
        )
        agent = self.build_agent(model, actions, nb_steps=nb_steps)

        if self._agent_value == "DQN" or self._agent_value == "SARSA":
            agent.compile(
                Adam(
                    learning_rate=self._trial.suggest_float(
                        "adam_learning_rate", 0.00001, 0.1
                    ),
                    epsilon=self._trial.suggest_float("adam_epsilon", 1e-9, 1e-1),
                ),
                metrics=["mae", "accuracy"],
            )
        elif self._agent_value == "CEM":
            agent.compile()

        if self._load:
            print(f"Load existing model and train: path={load_path}/weights.h5f")
            agent.load_weights(f"{load_path}/weights.h5f")

        callbacks = []
        callbacks += [
            optuna.integration.KerasPruningCallback(
                self._trial, "episode_reward", interval=10_000
            )
        ]

        history = agent.fit(
            env,
            nb_steps=nb_steps,
            verbose=1,
            visualize=False,
            # action_repetition=2,
            log_interval=50_000,
            callbacks=callbacks,
        )

        return agent, history

    def validate_model(self, agent, env):
        scores = agent.test(env, nb_episodes=100, visualize=False)

        episode_reward = np.mean(scores.history["episode_reward"])
        nb_steps = np.mean(scores.history["nb_steps"])
        print(f"episode_reward: {episode_reward}")
        print(f"nb_steps: {nb_steps}")

        return episode_reward

    def train(self, nb_steps=10_000, load_path="", env_config="", name=""):
        return self._train(
            nb_steps=nb_steps,
            load_path=load_path,
            env_config=env_config,
        )

    def _train(self, nb_steps=10_000, load_path="", env_config="", name=""):
        date_start = dt.today()
        env = KniffelEnv(env_config, config_file_path=self._config_path)

        actions = env.action_space.n

        agent, train_score = self.train_agent(
            actions=actions,
            env=env,
            nb_steps=nb_steps,
            load_path=load_path,
        )

        test_scores = self.validate_model(agent, env=env)

        return test_scores

    def predict_and_apply(self, agent, kniffel: Kniffel, state, logging=False):
        action = agent.forward(state)

        enum_action = EnumAction(action)

        if logging:
            print(f"      Action: {enum_action}")

        if EnumAction.FINISH_ONES is enum_action:
            kniffel.finish_turn(KniffelOptions.ONES)
        if EnumAction.FINISH_TWOS is enum_action:
            kniffel.finish_turn(KniffelOptions.TWOS)
        if EnumAction.FINISH_THREES is enum_action:
            kniffel.finish_turn(KniffelOptions.THREES)
        if EnumAction.FINISH_FOURS is enum_action:
            kniffel.finish_turn(KniffelOptions.FOURS)
        if EnumAction.FINISH_FIVES is enum_action:
            kniffel.finish_turn(KniffelOptions.FIVES)
        if EnumAction.FINISH_SIXES is enum_action:
            kniffel.finish_turn(KniffelOptions.SIXES)
        if EnumAction.FINISH_THREE_TIMES is enum_action:
            kniffel.finish_turn(KniffelOptions.THREE_TIMES)
        if EnumAction.FINISH_FOUR_TIMES is enum_action:
            kniffel.finish_turn(KniffelOptions.FOUR_TIMES)
        if EnumAction.FINISH_FULL_HOUSE is enum_action:
            kniffel.finish_turn(KniffelOptions.FULL_HOUSE)
        if EnumAction.FINISH_SMALL_STREET is enum_action:
            kniffel.finish_turn(KniffelOptions.SMALL_STREET)
        if EnumAction.FINISH_LARGE_STREET is enum_action:
            kniffel.finish_turn(KniffelOptions.LARGE_STREET)
        if EnumAction.FINISH_KNIFFEL is enum_action:
            kniffel.finish_turn(KniffelOptions.KNIFFEL)
        if EnumAction.FINISH_CHANCE is enum_action:
            kniffel.finish_turn(KniffelOptions.CHANCE)

        # Continue enum_actions
        if EnumAction.NEXT_0 is enum_action:
            kniffel.add_turn(keep=[0, 0, 0, 0, 0])
        if EnumAction.NEXT_1 is enum_action:
            kniffel.add_turn(keep=[0, 0, 0, 0, 1])
        if EnumAction.NEXT_2 is enum_action:
            kniffel.add_turn(keep=[0, 0, 0, 1, 0])
        if EnumAction.NEXT_3 is enum_action:
            kniffel.add_turn(keep=[0, 0, 0, 1, 1])
        if EnumAction.NEXT_4 is enum_action:
            kniffel.add_turn(keep=[0, 0, 1, 0, 0])
        if EnumAction.NEXT_5 is enum_action:
            kniffel.add_turn(keep=[0, 0, 1, 0, 1])
        if EnumAction.NEXT_6 is enum_action:
            kniffel.add_turn(keep=[0, 0, 1, 1, 0])
        if EnumAction.NEXT_7 is enum_action:
            kniffel.add_turn(keep=[0, 0, 1, 1, 1])
        if EnumAction.NEXT_8 is enum_action:
            kniffel.add_turn(keep=[0, 1, 0, 0, 0])
        if EnumAction.NEXT_9 is enum_action:
            kniffel.add_turn(keep=[0, 1, 0, 0, 1])
        if EnumAction.NEXT_10 is enum_action:
            kniffel.add_turn(keep=[0, 1, 0, 1, 0])
        if EnumAction.NEXT_11 is enum_action:
            kniffel.add_turn(keep=[0, 1, 0, 1, 1])
        if EnumAction.NEXT_12 is enum_action:
            kniffel.add_turn(keep=[0, 1, 1, 0, 0])
        if EnumAction.NEXT_13 is enum_action:
            kniffel.add_turn(keep=[0, 1, 1, 0, 1])
        if EnumAction.NEXT_14 is enum_action:
            kniffel.add_turn(keep=[0, 1, 1, 1, 0])
        if EnumAction.NEXT_15 is enum_action:
            kniffel.add_turn(keep=[0, 1, 1, 1, 1])
        if EnumAction.NEXT_16 is enum_action:
            kniffel.add_turn(keep=[1, 0, 0, 0, 0])
        if EnumAction.NEXT_17 is enum_action:
            kniffel.add_turn(keep=[1, 0, 0, 0, 1])
        if EnumAction.NEXT_18 is enum_action:
            kniffel.add_turn(keep=[1, 0, 0, 1, 0])
        if EnumAction.NEXT_19 is enum_action:
            kniffel.add_turn(keep=[1, 0, 0, 1, 1])
        if EnumAction.NEXT_20 is enum_action:
            kniffel.add_turn(keep=[1, 0, 1, 0, 0])
        if EnumAction.NEXT_21 is enum_action:
            kniffel.add_turn(keep=[1, 0, 1, 0, 1])
        if EnumAction.NEXT_22 is enum_action:
            kniffel.add_turn(keep=[1, 0, 1, 1, 0])
        if EnumAction.NEXT_23 is enum_action:
            kniffel.add_turn(keep=[1, 0, 1, 1, 1])
        if EnumAction.NEXT_24 is enum_action:
            kniffel.add_turn(keep=[1, 1, 0, 0, 0])
        if EnumAction.NEXT_25 is enum_action:
            kniffel.add_turn(keep=[1, 1, 0, 0, 1])
        if EnumAction.NEXT_26 is enum_action:
            kniffel.add_turn(keep=[1, 1, 0, 1, 0])
        if EnumAction.NEXT_27 is enum_action:
            kniffel.add_turn(keep=[1, 1, 0, 1, 1])
        if EnumAction.NEXT_28 is enum_action:
            kniffel.add_turn(keep=[1, 1, 1, 0, 0])
        if EnumAction.NEXT_29 is enum_action:
            kniffel.add_turn(keep=[1, 1, 1, 0, 1])
        if EnumAction.NEXT_30 is enum_action:
            kniffel.add_turn(keep=[1, 1, 1, 1, 0])
        if EnumAction.NEXT_31 is enum_action:
            kniffel.add_turn(keep=[1, 1, 1, 1, 1])

        if EnumAction.FINISH_ONES_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.ONES_SLASH)
        if EnumAction.FINISH_TWOS_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.TWOS_SLASH)
        if EnumAction.FINISH_THREES_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.THREES_SLASH)
        if EnumAction.FINISH_FOURS_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.FOURS_SLASH)
        if EnumAction.FINISH_FIVES_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.FIVES_SLASH)
        if EnumAction.FINISH_SIXES_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.SIXES_SLASH)
        if EnumAction.FINISH_THREE_TIMES_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.THREE_TIMES_SLASH)
        if EnumAction.FINISH_FOUR_TIMES_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.FOUR_TIMES_SLASH)
        if EnumAction.FINISH_FULL_HOUSE_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.FULL_HOUSE_SLASH)
        if EnumAction.FINISH_SMALL_STREET_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.SMALL_STREET_SLASH)
        if EnumAction.FINISH_LARGE_STREET_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.LARGE_STREET_SLASH)
        if EnumAction.FINISH_KNIFFEL_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.KNIFFEL_SLASH)
        if EnumAction.FINISH_CHANCE_SLASH is enum_action:
            kniffel.finish_turn(KniffelOptions.CHANCE_SLASH)

    def test(self, agent):
        points = []
        break_counter = 0

        for _ in range(self._test_episodes):
            kniffel = Kniffel()
            while True:
                try:
                    state = kniffel.get_state()
                    self.predict_and_apply(agent, kniffel, state)
                except BaseException as e:
                    if e == ex.GameFinishedException:
                        points.append(kniffel.get_points())
                        break
                    else:
                        points.append(kniffel.get_points())
                        break_counter += 1
                        break

                points.append(kniffel.get_points())

        return break_counter, mean(points), max(points), min(points)


def objective(trial):
    base_hp = {
        "windows_length": [1],  # range(1, 3),
        "batch_size": [32],
        "target_model_update": [
            0.00001,
            0.0005,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
            50,
            100,
            200,
            300,
            400,
            500,
            750,
            1000,
            2_500,
            5_000,
            7_500,
            10_000,
            15_000,
        ],
        "dueling_option": ["avg"],
        "activation": ["linear", "softmax", "sigmoid"],
        "enable_double_dqn": [True, False],
        "agent": ["DQN", "CEM", "SARSA"],
        "train_policy": [
            "LinearAnnealedPolicy",
            "EpsGreedyQPolicy",
            "GreedyQPolicy",
            "BoltzmannQPolicy",
            "MaxBoltzmannQPolicy",
            "BoltzmannGumbelQPolicy",
        ],
        "test_policy": [
            "LinearAnnealedPolicy",
            "EpsGreedyQPolicy",
            "GreedyQPolicy",
            "BoltzmannQPolicy",
            "MaxBoltzmannQPolicy",
        ],
    }

    env_config = {
        "reward_step": 0,
        "reward_roll_dice": 0.5,
        "reward_game_over": -200,
        "reward_slash": -10,
        "reward_bonus": 20,
        "reward_finish": 50,
    }

    ai = KniffelAI(
        load=False,
        hyperparater_base=base_hp,
        config_path="src/ai/Kniffel.CSV",
        path_prefix="",
        trial=trial,
    )

    score = ai.train(env_config=env_config, nb_steps=50_000)
    return score


def optuna_func():
    study = optuna.load_study(
        study_name=_study_name,
        storage=f"mysql://kniffel:{_pw}@kniffel-do-user-12010256-0.b.db.ondigitalocean.com:25060/kniffel",
    )

    study.optimize(objective, n_trials=100)


import concurrent.futures

_pw = ""
_study_name = ""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    pw = [s for s in sys.argv if s.startswith("p=")][0].split("=")[1]
    study_name = [s for s in sys.argv if s.startswith("name=")][0].split("=")[1]

    _pw = pw
    _study_name = study_name

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"mysql://kniffel:{pw}@kniffel-do-user-12010256-0.b.db.ondigitalocean.com:25060/kniffel",
    )

    with concurrent.futures.ProcessPoolExecutor() as executor:
        process1 = executor.submit(optuna_func)
        process2 = executor.submit(optuna_func)
        process3 = executor.submit(optuna_func)
        process4 = executor.submit(optuna_func)

    # print("Number of finished trials: {}".format(len(study.trials)))

    # print("Best trial:")
    # trial = study.best_trial

    # print("  Value: {}".format(trial.value))

    # print("  Params: ")
    # for key, value in trial.params.items():
    #    print("    {}: {}".format(key, value))

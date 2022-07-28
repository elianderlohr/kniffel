# Standard imports
from statistics import mean
from datetime import datetime as dt
from tokenize import Triple
import warnings
import numpy as np
import json
import optuna
import os
from pathlib import Path
import sys

# Keras / Tensorflow imports
import tensorflow as tf

from rl.agents import DQNAgent, CEMAgent, SARSAAgent
from rl.policy import (
    BoltzmannQPolicy,
    EpsGreedyQPolicy,
    LinearAnnealedPolicy,
    SoftmaxPolicy,
    GreedyQPolicy,
    MaxBoltzmannQPolicy,
    BoltzmannGumbelQPolicy,
)
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


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

    def _return_trial(self, key):
        return self._trial.suggest_categorical(key, self._hyperparater_base[key])

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

        for i in range(1, self._return_trial("layers") + 1):
            model.add(
                Dense(
                    self._return_trial("n_units_l" + str(i)),
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

        elif key == "SoftmaxPolicy":

            policy = SoftmaxPolicy()

        elif key == "EpsGreedyQPolicy":

            policy = EpsGreedyQPolicy(
                eps=self._return_trial(
                    "adam_epsilon",
                )
            )

        elif key == "GreedyQPolicy":

            policy = GreedyQPolicy()

        elif key == "BoltzmannQPolicy":

            policy = BoltzmannQPolicy()

        elif key == "MaxBoltzmannQPolicy":

            policy = MaxBoltzmannQPolicy(
                eps=self._return_trial(
                    "adam_epsilon",
                )
            )
        elif key == "BoltzmannGumbelQPolicy":

            policy = BoltzmannGumbelQPolicy()

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
        train_policy = None
        policy = None

        memory = SequentialMemory(
            limit=500_000,
            window_length=self._return_trial("windows_length"),
        )

        agent_value = self._return_trial("agent")

        if agent_value == "DQN":
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

        elif agent_value == "CEM":
            agent = CEMAgent(
                model=model,
                memory=memory,
                nb_actions=actions,
                nb_steps_warmup=1_000,
                batch_size=self._return_trial("batch_size"),
            )

        elif agent_value == "SARSA":
            agent = SARSAAgent(
                model=model,
                policy=self.get_policy(self._return_trial("train_policy")),
                test_policy=self.get_policy(self._return_trial("test_policy")),
                nb_actions=actions,
                nb_steps_warmup=1_000,
                gamma=self._return_trial("gamma"),
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

        agent.compile(
            Adam(
                learning_rate=self._return_trial(
                    "adam_learning_rate",
                ),
                epsilon=self._return_trial(
                    "adam_epsilon",
                ),
            ),
            metrics=["mae", "accuracy"],
        )

        if self._load:
            print(f"Load existing model and train: path={load_path}/weights.h5f")
            agent.load_weights(f"{load_path}/weights.h5f")

        history = agent.fit(
            env,
            nb_steps=nb_steps,
            verbose=1,
            visualize=False,
            # action_repetition=2,
            log_interval=10_000,
        )

        return agent, history

    def validate_model(self, agent, env):
        scores = agent.test(env, nb_episodes=100, visualize=False)

        episode_reward = np.mean(scores.history["episode_reward"])
        nb_steps = np.mean(scores.history["nb_steps"])
        print(f"episode_reward: {episode_reward}")
        print(f"nb_steps: {nb_steps}")

        return nb_steps

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

    # Use Model
    def play(self, path, episodes, env_config, random=False, logging=False):
        if random:
            self.play_random(episodes, env_config)
        else:
            self.use_model(path, episodes, env_config, logging=logging)

    def play_random(self, episodes, env_config):
        env = KniffelEnv(env_config, logging=True, config_file_path=self._config_path)

        round = 1
        for episode in range(1, episodes + 1):
            state = env.reset()
            done = False
            score = 0

            print("####")
            print(round)

            try_out = 1
            while not done:
                action = env.action_space.sample()
                n_state, reward, done, info = env.step(action)
                score += reward
                print(f"Try: {try_out}")
                print(f"Action: {action}")
                print(f"Score: {score}")
                print(n_state)
                try_out += 1

            print("Episode:{} Score:{}".format(episode, score))

            round += 1

    def use_model(self, path, episodes, env_config, logging=False):

        env = KniffelEnv(
            env_config, logging=logging, config_file_path=self._config_path
        )

        f = open(f"{path}/configuration.json")
        hyperparameter = dict(json.load(f))

        actions = env.action_space.n

        model = self.build_model(
            actions, hyperparameter, hyperparameter["window_length"]
        )

        agent = self.build_agent(
            model, actions, nb_steps=episodes, hyperparameter=hyperparameter
        )

        agent.compile(
            optimizer=Adam(
                learning_rate=self._trial.suggest_categorical(
                    "adam_learning_rate", hyperparameter["adam_learning_rate"]
                )
            ),
            metrics=["mae"],
        )

        agent.load_weights(f"{path}/weights.h5f")

        points = []
        rounds = []
        break_counter = 0
        for e in range(episodes):
            if logging:
                print(f"Game: {e}")

            kniffel = Kniffel()
            rounds_counter = 1
            while True:
                state = kniffel.get_state()
                if logging:
                    print(f"    Round: {rounds_counter}")

                try:
                    self.predict_and_apply(agent, kniffel, state, logging)
                    rounds_counter += 1
                    if logging:
                        print(f"    State: {state}")
                        print(f"       Points: {kniffel.get_points()}")
                        print(f"       Prediction Allowed: True")
                except BaseException as e:
                    if e == ex.GameFinishedException:
                        points.append(kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        if logging:
                            print("       Prediction Allowed: False")
                            print("       Game Finished: True")

                        break
                    else:
                        points.append(kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        if logging:
                            print("       Prediction Allowed: False")

                        break

        print()
        print(f"Finished games: {episodes - break_counter}")
        print(f"Average points: {mean(points)}")
        print(f"Max points: {max(points)}")
        print(f"Min points: {min(points)}")
        print(f"Average rounds: {mean(rounds)}")
        print(f"Max rounds: {max(rounds)}")
        print(f"Min rounds: {min(rounds)}")

        return break_counter, sum(points) / len(points), max(points), min(points)


def objective(trial):
    base_hp = {
        "windows_length": range(1, 3),
        "adam_learning_rate": [
            0.00001,
            0.0005,
            0.0001,
            0.0005,
            0.001,
            0.005,
            0.01,
            0.05,
            0.1,
        ],
        "adam_epsilon": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        "batch_size": [32, 64, 128, 512, 1028, 1512, 2056],
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
        "layers": [4, 3, 2, 1],
        "n_units_l1": [256, 128, 96, 64, 32, 16],
        "n_units_l2": [256, 128, 96, 64, 32, 16],
        "n_units_l3": [256, 128, 96, 64, 32, 16],
        "n_units_l4": [256, 128, 96, 64, 32, 16],
        "enable_double_dqn": [True, False],
        "agent": ["DQN", "CEM", "SARSA"],
        "gamma": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0, 9],
        "train_policy": [
            "LinearAnnealedPolicy",
            "SoftmaxPolicy",
            "SoftmaxPolicy",
            "EpsGreedyQPolicy",
            "GreedyQPolicy",
            "BoltzmannQPolicy",
            "MaxBoltzmannQPolicy",
            "BoltzmannGumbelQPolicy",
        ],
        "test_policy": [
            "LinearAnnealedPolicy",
            "SoftmaxPolicy",
            "SoftmaxPolicy",
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

    try:
        score = ai.train(env_config=env_config, nb_steps=50_000)
        return score
    except:
        return 0


def optuna_func():
    study = optuna.load_study(
        study_name=_study_name,
        storage=f"mysql://kniffel:{_pw}@kniffel-do-user-12010256-0.b.db.ondigitalocean.com:25060/kniffel",
    )

    study.optimize(objective, n_trials=100)


def runInParallel(*fns):
    proc = []

    for fn in fns:
        p = Process(target=fn)
        p.start()
        proc.append(p)

    for p in proc:
        p.join()


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

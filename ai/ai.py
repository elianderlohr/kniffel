from math import gamma
from pickletools import optimize
from statistics import mean
from datetime import datetime as dt
from sympy import N, use

import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import (
    BoltzmannQPolicy,
    BoltzmannGumbelQPolicy,
    LinearAnnealedPolicy,
    EpsGreedyQPolicy,
)
from rl.memory import SequentialMemory

import numpy as np
import os
import sys
import inspect

import warnings
import json
import itertools
import matplotlib.pyplot as plt

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
)

from kniffel.classes.options import KniffelOptions
from kniffel.classes.kniffel import Kniffel
from env import EnumAction
from env import KniffelEnv

import rl.callbacks
from rl.callbacks import FileLogger, ModelIntervalCheckpoint


class EpisodeLogger(rl.callbacks.Callback):
    def __init__(self):
        self.observations = {}
        self.rewards = {}
        self.actions = {}

    def on_episode_begin(self, episode, logs):
        self.observations[episode] = []
        self.rewards[episode] = []
        self.actions[episode] = []

    def on_step_end(self, step, logs):
        episode = logs["episode"]
        self.observations[episode].append(logs["observation"])
        self.rewards[episode].append(logs["reward"])
        self.actions[episode].append(logs["action"])


class KniffelAI:
    config = {
        "windows_length": [1],
        "adam_learning_rate": [0.0001],
        "batch_size": [20, 32],
        "target_model_update": [0.0001],
        "dueling_option": ["avg"],
        "eps": [0.1, 0.2, 0.3, 0.4, 0.5],
    }

    # Model

    def build_model(self, actions, windows_length):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=(windows_length, 13, 16)))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Flatten())
        model.add(Dense(256, activation="relu"))
        model.add(Dense(actions, activation="softmax"))
        return model

    def build_agent(
        self,
        model,
        actions,
        windows_length,
        batch_size,
        target_model_update,
        dueling_option,
        eps,
        nb_steps,
    ):
        # policy = BoltzmannQPolicy()
        # policy = BoltzmannGumbelQPolicy()
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr="eps",
            value_max=1.0,
            value_min=0.1,
            value_test=0.05,
            nb_steps=10000,  # nb_steps,
        )
        memory = SequentialMemory(
            limit=1000000,
            window_length=windows_length,
        )
        dqn = DQNAgent(
            model=model,
            memory=memory,
            policy=policy,
            nb_actions=actions,
            nb_steps_warmup=10000,
            target_model_update=target_model_update,
            batch_size=batch_size,
            dueling_type=dueling_option,
        )
        return dqn

    # Train models by applying config
    def grid_search_test(self, nb_steps=20000):
        result = list(
            itertools.product(
                self.config["windows_length"],
                self.config["adam_learning_rate"],
                self.config["batch_size"],
                self.config["target_model_update"],
                self.config["dueling_option"],
                self.config["eps"],
            )
        )

        datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")
        path = f"configuration/p_date={datetime}"

        os.mkdir(path)

        header = "duration;nb_steps;windows_length;adam_learning_rate;batch_size;target_model_update;mean_episode;max_episode;min_episode;mean;max;min;break_counter;n;dueling_type;eps\n"
        with open(f"{path}/csv_configuration.csv", "a") as file:
            file.write(header)
            file.close()

        i = 1
        for (
            windows_length,
            adam_learning_rate,
            batch_size,
            target_model_update,
            dueling_option,
            eps,
        ) in result:
            print("#################")
            print(f"Test {i} from {len(result)}")

            hyperparameter = {
                "windows_length": windows_length,
                "adam_learning_rate": adam_learning_rate,
                "batch_size": batch_size,
                "target_model_update": target_model_update,
                "dueling_option": dueling_option,
                "eps": eps,
            }

            content, csv = self.train(
                hyperparameter=hyperparameter,
                nb_steps=nb_steps,
            )

            try:

                with open(f"{path}/text_configuration.txt", "a") as file:
                    file.write(content)
                    file.close()

                with open(f"{path}/csv_configuration.csv", "a") as file:
                    file.write(csv)
                    file.close()
            except Exception as e:
                print(e)

            i = i + 1

    def train_dqn(
        self,
        actions,
        hyperparameter,
        env,
        nb_steps,
        callbacks,
        save=False,
        load=False,
        load_path="",
    ):

        windows_length = hyperparameter["windows_length"]
        adam_learning_rate = hyperparameter["adam_learning_rate"]
        batch_size = hyperparameter["batch_size"]
        target_model_update = hyperparameter["target_model_update"]
        dueling_option = hyperparameter["dueling_option"]
        eps = hyperparameter["eps"]

        model = self.build_model(actions, windows_length)
        dqn = self.build_agent(
            model,
            actions,
            windows_length,
            batch_size=batch_size,
            target_model_update=target_model_update,
            dueling_option=dueling_option,
            eps=eps,
            nb_steps=nb_steps,
        )

        dqn.compile(Adam(learning_rate=adam_learning_rate), metrics=["mae"])

        if load:
            print(f"Load existing model and train: path={load_path}/weights.h5f")
            dqn.load_weights(f"{load_path}/weights.h5f")

        if save:
            dqn.fit(
                env, nb_steps=nb_steps, verbose=1, visualize=False, callbacks=callbacks
            )
        else:
            dqn.fit(env, nb_steps=nb_steps, verbose=1, visualize=False)

        return dqn

    def validate_model(self, dqn, env):
        scores = dqn.test(env, nb_episodes=100, visualize=False)

        print(np.mean(scores.history["episode_reward"]))
        _ = dqn.test(env, nb_episodes=15, visualize=False)

        return scores

    def get_configuration(self, dqn, scores, date_start, hyperparameter, nb_steps, n):
        break_counter, mean, max, min = self.test(dqn=dqn, n=n)
        date_end = dt.today()
        datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")

        duration = date_end - date_start

        mean_episode = str(np.mean(scores.history["episode_reward"]))
        max_episode = str(np.max(scores.history["episode_reward"]))
        min_episode = str(np.min(scores.history["episode_reward"]))

        content = self.get_configuration_as_text(
            hyperparameter,
            datetime,
            duration,
            nb_steps,
            mean_episode,
            max_episode,
            min_episode,
            mean,
            max,
            min,
            break_counter,
            n,
        )

        windows_length = hyperparameter["windows_length"]
        adam_learning_rate = hyperparameter["adam_learning_rate"]
        batch_size = hyperparameter["batch_size"]
        target_model_update = hyperparameter["target_model_update"]
        dueling_option = hyperparameter["dueling_option"]
        eps = hyperparameter["eps"]

        csv = f"{duration.total_seconds()};{nb_steps};{windows_length};{adam_learning_rate};{batch_size};{target_model_update};{mean_episode};{max_episode};{min_episode};{mean};{max};{min};{break_counter};{n};{dueling_option};{eps}\n"

        return content, csv

    def train(
        self,
        hyperparameter,
        nb_steps=10000,
        load=False,
        load_path="",
        save=False,
    ):
        date_start = dt.today()
        env = KniffelEnv()

        actions = env.action_space.n

        # parameter
        n = 100
        callbacks = []

        if save:
            datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")
            path = f"weights/p_date={datetime}"

            # Create dir
            os.mkdir(path)

            checkpoint_weights_filename = path + "/dqn_weights_{step}.h5f"
            log_filename = path + "/dqn_log.json"

            callbacks = [
                ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)
            ]
            callbacks += [FileLogger(log_filename, interval=100)]

        dqn = self.train_dqn(
            actions=actions,
            hyperparameter=hyperparameter,
            env=env,
            nb_steps=nb_steps,
            load=load,
            load_path=load_path,
            callbacks=callbacks,
            save=save,
        )

        scores = self.validate_model(dqn=dqn, env=env)

        content, csv = self.get_configuration(
            dqn=dqn,
            scores=scores,
            date_start=date_start,
            hyperparameter=hyperparameter,
            nb_steps=nb_steps,
            n=n,
        )

        if save:
            # save weights and configuration as json
            dqn.save_weights(f"{path}/weights.h5f", overwrite=False)

            json_object = json.dumps(hyperparameter, indent=4)

            with open(f"{path}/configuration.json", "w") as file:
                file.write(json_object)
                file.close()

            with open(f"{path}/configuration.txt", "w") as file:
                file.write(content)
                file.close()

        return content, csv

    def get_configuration_as_text(
        self,
        hyperparameter,
        datetime,
        duration,
        nb_steps,
        mean_episode,
        max_episode,
        min_episode,
        mean,
        max,
        min,
        break_counter,
        n,
    ):

        return (
            f"""
####################
TEST RUN

date: {datetime}
duration in seconds: {duration.total_seconds()}
nb_steps: {nb_steps}

HYPERPARAMETER

    windows_length: {hyperparameter["windows_length"]}
    adam_learning_rate: {hyperparameter["adam_learning_rate"]}
    
    batch_size: {hyperparameter["batch_size"]}
    target_model_update: {hyperparameter["target_model_update"]}

    dueling_option: {hyperparameter["dueling_option"]}
    eps: {hyperparameter["eps"]}
RESULT

    TRAIN:
        AVG Reward: {mean_episode}
        Max Reward: {max_episode}
        Min Reward: {min_episode}

    TEST:
        AVG Score: {mean}
        Max Score: {max}
        Min Score: {min}
        Breakcounter: {break_counter}/{n}
        """.rstrip()
            .lstrip()
            .strip()
        )

    def predict_and_apply(self, dqn: DQNAgent, kniffel: Kniffel, state):
        action = dqn.forward(state)
        enum_action = EnumAction(action)

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

    def test(self, dqn, n):
        env = KniffelEnv()

        points = []
        break_counter = 0

        for i in range(n):
            kniffel = Kniffel()
            while True:
                try:
                    state = kniffel.get_array()
                    self.predict_and_apply(dqn, kniffel, state)
                except Exception as e:
                    points.append(kniffel.get_points())
                    break_counter += 1
                    break

                points.append(kniffel.get_points())

        return break_counter, mean(points), max(points), min(points)

    # Use Model

    def play(self, path, episodes, random=False):
        if random:
            random(episodes)
        else:
            self.use_model(path, episodes)

    def random(self, episodes):
        env = KniffelEnv()

        for episode in range(1, episodes + 1):
            state = env.reset()
            done = False
            score = 0

            while not done:
                action = env.action_space.sample()
                n_state, reward, done, info = env.step(action)
                score += reward
                print(n_state)

            print("Episode:{} Score:{}".format(episode, score))

    def use_model(
        self,
        path,
        episodes,
    ):
        env = KniffelEnv()

        f = open(f"{path}/configuration.json")
        data = json.load(f)

        windows_length = data["windows_length"]
        batch_size = data["batch_size"]
        target_model_update = data["target_model_update"]
        dueling_option = data["dueling_option"]
        adam_learning_rate = data["adam_learning_rate"]
        eps = data["eps"]

        actions = env.action_space.n

        model = self.build_model(actions, windows_length)
        dqn = self.build_agent(
            model,
            actions,
            windows_length,
            batch_size=batch_size,
            target_model_update=target_model_update,
            dueling_option=dueling_option,
            eps=eps,
            nb_steps=episodes,
        )
        dqn.compile(Adam(learning_rate=adam_learning_rate), metrics=["mae"])

        dqn.load_weights(f"{path}/weights.h5f")

        points = []
        break_counter = 0

        for i in range(episodes):
            kniffel = Kniffel()
            while True:
                try:
                    state = kniffel.get_array()
                    self.predict_and_apply(dqn, kniffel, state)
                except Exception as e:
                    points.append(kniffel.get_points())
                    break_counter += 1
                    break

                points.append(kniffel.get_points())

        print()
        print(f"break_counter: {break_counter}")
        print(f"AVG. {sum(points) / len(points)}")
        print(f"MAX: {max(points)}")
        print(f"MIN: {min(points)}")

        return break_counter, sum(points) / len(points), max(points), min(points)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    ai = KniffelAI()

    ai.play(path="weights/p_date=2022-04-10-17_25_53", episodes=10000)
    # ai.grid_search_test(nb_steps=20000)

    hyperparameter = {
        "windows_length": 1,
        "adam_learning_rate": 0.0001,
        "batch_size": 20,
        "target_model_update": 0.0001,
        "dueling_option": "avg",
        "eps": 0.5,
    }
    """
    ai.train(
        hyperparameter=hyperparameter,
        load=True,
        load_path="weights\p_date=2022-04-08-22_14_23",
        nb_steps=1000000,
        save=True,
    )"""

# Standard imports
from statistics import mean
from datetime import datetime as dt
import numpy as np
import os
import sys
import inspect
import warnings
import json

# Keras / Tensorflow imports
import tensorflow as tf

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Kniffel imports
sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
)

from kniffel.classes.options import KniffelOptions
from kniffel.classes.kniffel import Kniffel
from ai.hyperparameter import Hyperparameter
from env import EnumAction
from env import KniffelEnv


class KniffelAI:
    # Save model
    _save = False

    # Load model from path
    _load = False

    # Hyperparameter object
    _hp = None

    # Test episodes
    _test_episodes = 100

    # path prefix
    _path_prefix = ""

    def __init__(
        self,
        save=False,
        load=False,
        predefined_layers=False,
        test_episodes=100,
        path_prefix="",
        hyperparater_base={},
    ):
        self._save = save
        self._load = load
        self._hp = Hyperparameter(
            randomize=True,
            predefined_layers=predefined_layers,
            base_hp=hyperparater_base,
        )
        self._test_episodes = test_episodes

        if path_prefix == "":
            try:
                import google.colab

                self._path_prefix = "/"
            except:
                self._path_prefix = ""
        else:
            self._path_prefix = path_prefix

    # Model
    def build_model(self, actions, hyperparameter):
        model = tf.keras.Sequential()
        model.add(Flatten(input_shape=(hyperparameter["windows_length"], 1, 41)))

        for i in range(1, hyperparameter["layers"] + 1):
            model.add(Dense(hyperparameter["unit_" + str(i)], activation="relu"))

        model.add(Dense(actions, activation=hyperparameter["activation"]))
        model.summary()
        return model

    def build_agent(self, model, actions, nb_steps, hyperparameter):
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
            window_length=hyperparameter["windows_length"],
        )

        train_policy = EpsGreedyQPolicy()  # BoltzmannQPolicy()

        agent = DQNAgent(
            model=model,
            memory=memory,
            policy=train_policy,
            nb_actions=actions,
            nb_steps_warmup=1_000,
            target_model_update=hyperparameter["target_model_update"],
            batch_size=hyperparameter["batch_size"],
            dueling_type=hyperparameter["dueling_option"],
        )

        return agent

    # Train models by applying config
    def grid_search_test(self, nb_steps=20_000, env_config={}):
        datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")
        path = f"{self._path_prefix}configuration/p_date={datetime}"

        hyperparameter_csv = ";".join(
            str(e) for e in list(dict(self._hp.get()[0]).keys())
        )
        print(hyperparameter_csv)
        self._append_file(
            f"{path}/csv_configuration.csv",
            content=f"duration;nb_steps;mean_train;max_train;min_train;mean_test_agent;max_test_agent;min_test_agent;mean_test_own;max_test_own;min_test_own;break_counter;n;{hyperparameter_csv}\n",
        )

        i = 1
        for hyperparameter in self._hp.get():
            print()
            print("#################")
            print(f"Test {i} from {len(self._hp.get())}")
            print()
            print(hyperparameter)
            print()

            csv = self.train(
                hyperparameter=hyperparameter, nb_steps=nb_steps, env_config=env_config
            )

            self._append_file(f"{path}/csv_configuration.csv", content=csv)

            i = i + 1

    def _append_file(self, path, content, retry=0):
        try:
            with open(path, "a") as file:
                file.write(content)
                file.close()
        except:
            try:
                os.mkdir(os.path.dirname(path))
                if retry < 3:
                    self._append_file(path, content, retry + 1)
            except Exception as e:
                print(path)
                print(e)

    def train_agent(
        self,
        actions,
        hyperparameter,
        env,
        nb_steps,
        callbacks,
        load_path="",
    ):
        model = self.build_model(actions, hyperparameter)
        agent = self.build_agent(
            model,
            actions,
            nb_steps=nb_steps,
            hyperparameter=hyperparameter,
        )

        agent.compile(
            Adam(
                learning_rate=hyperparameter["adam_learning_rate"],
                epsilon=hyperparameter["adam_epsilon"],
            ),
            metrics=["mae"],
        )

        if self._load:
            print(f"Load existing model and train: path={load_path}/weights.h5f")
            agent.load_weights(f"{load_path}/weights.h5f")

        if self._save:
            history = agent.fit(
                env, nb_steps=nb_steps, verbose=1, visualize=False, callbacks=callbacks
            )
        else:
            history = agent.fit(env, nb_steps=nb_steps, verbose=1, visualize=False)

        return agent, history

    def validate_model(self, agent, env):
        scores = agent.test(env, nb_episodes=100, visualize=False)

        # print(np.mean(scores.history["episode_reward"]))
        # _ = agent.test(env, nb_episodes=15, visualize=False)

        return scores

    def get_configuration(
        self, agent, train_scores, test_scores, date_start, hyperparameter, nb_steps
    ):
        break_counter, mean_own, max_own, min_own = self.test(agent)
        date_end = dt.today()

        duration = date_end - date_start

        mean_train = str(np.mean(train_scores.history["episode_reward"]))
        max_train = str(np.max(train_scores.history["episode_reward"]))
        min_train = str(np.min(train_scores.history["episode_reward"]))

        mean_test = str(np.mean(test_scores.history["episode_reward"]))
        max_test = str(np.max(test_scores.history["episode_reward"]))
        min_test = str(np.min(test_scores.history["episode_reward"]))

        hyperparameter_csv = ";".join(
            str(e) for e in list(dict(hyperparameter).values())
        )

        csv = f"{duration.total_seconds()};{nb_steps};{mean_train};{max_train};{min_train};{mean_test};{max_test};{min_test};{mean_own};{max_own};{min_own};{break_counter};{self._test_episodes};{hyperparameter_csv}\n"

        return csv

    def train(self, hyperparameter, nb_steps=10_000, load_path="", env_config=""):
        date_start = dt.today()
        env = KniffelEnv(env_config)

        actions = env.action_space.n

        callbacks = []

        if self._save:
            datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")
            path = f"{self._path_prefix}weights/p_date={datetime}"

            # Create dir
            os.mkdir(path)

            checkpoint_weights_filename = path + "/agent_weights_{step}.h5f"
            log_filename = path + "/agent_log.json"

            callbacks = [
                ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)
            ]
            callbacks += [FileLogger(log_filename, interval=100)]

        agent, train_score = self.train_agent(
            actions=actions,
            hyperparameter=hyperparameter,
            env=env,
            nb_steps=nb_steps,
            load_path=load_path,
            callbacks=callbacks,
        )

        test_scores = self.validate_model(agent, env=env)

        csv = self.get_configuration(
            agent=agent,
            train_scores=train_score,
            test_scores=test_scores,
            date_start=date_start,
            hyperparameter=hyperparameter,
            nb_steps=nb_steps,
        )

        if self._save:
            # save weights and configuration as json
            agent.save_weights(f"{path}/weights.h5f", overwrite=False)

            json_object = json.dumps(hyperparameter, indent=4)

            self._append_file(f"{path}/configuration.json", json_object)

        return csv

    def predict_and_apply(self, agent, kniffel: Kniffel, state):
        action = agent.forward(state)
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

    def test(self, agent):
        points = []
        break_counter = 0

        for i in range(self._test_episodes):
            kniffel = Kniffel()
            while True:
                try:
                    state = kniffel.get_array_v2()
                    self.predict_and_apply(agent, kniffel, state)
                except Exception as e:
                    points.append(kniffel.get_points())
                    break_counter += 1
                    break

                points.append(kniffel.get_points())

        return break_counter, mean(points), max(points), min(points)

    # Use Model

    def play(self, path, episodes, env_config, random=False):
        if random:
            self.play_random(episodes, env_config)
        else:
            self.use_model(path, episodes, env_config)

    def play_random(self, episodes, env_config):
        env = KniffelEnv(env_config)

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
                print(reward)
                try_out += 1

            print("Episode:{} Score:{}".format(episode, score))

            round += 1

    def use_model(self, path, episodes, env_config):
        env = KniffelEnv(env_config)

        f = open(f"{path}/configuration.json")
        hyperparameter = dict(json.load(f))

        actions = env.action_space.n

        model = self.build_model(actions, hyperparameter)
        agent = self.build_agent(
            model, actions, nb_steps=episodes, hyperparameter=hyperparameter
        )
        agent.compile(
            Adam(learning_rate=hyperparameter["adam_learning_rate"]), metrics=["mae"]
        )

        agent.load_weights(f"{path}/weights.h5f")

        points = []
        rounds = []
        break_counter = 0
        for _ in range(episodes):
            kniffel = Kniffel()
            rounds_counter = 0
            while True:
                try:
                    state = kniffel.get_array_v2()
                    self.predict_and_apply(agent, kniffel, state)
                    rounds_counter += 1
                except:
                    points.append(kniffel.get_points())
                    break_counter += 1
                    rounds_counter = 0
                    break

                rounds.append(rounds_counter)
                points.append(kniffel.get_points())

        print()
        print(f"break_counter: {break_counter}")
        print(f"AVG. {sum(points) / len(points)}")
        print(f"MAX: {max(points)}")
        print(f"MIN: {min(points)}")
        print(f"AVG rounds: {mean(rounds)}")
        print(f"Max rounds: {max(rounds)}")
        print(f"Min rounds: {min(rounds)}")

        return break_counter, sum(points) / len(points), max(points), min(points)


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
        "reward_game_over": -100,
        "reward_slash": -10,
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

    # ai.play(
    #    path="weights\p_date=2022-05-19-06_37_37", episodes=1_000, env_config=env_config
    # )

    # ai.grid_search_test(nb_steps=20_000, env_config=env_config)

    hyperparameter = {
        "windows_length": 1,
        "adam_learning_rate": 0.0009,
        "adam_epsilon": 1e-5,
        "batch_size": 32,
        "target_model_update": 0.0009,
        "dueling_option": "avg",
        "activation": "linear",
        "layers": 2,
        "unit_1": 32,
        "unit_2": 16,
    }

    ai.train(hyperparameter=hyperparameter, nb_steps=250_000, env_config=env_config)
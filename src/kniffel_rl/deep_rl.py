# Standard imports
from distutils.log import info
import json
import os
import sys
import warnings
from datetime import datetime as dt
from pathlib import Path
from statistics import mean
from progress.bar import IncrementalBar

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Keras imports
import rl
from rl.agents import CEMAgent, DQNAgent, SARSAAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.memory import EpisodeParameterMemory, SequentialMemory
from rl.policy import (
    BoltzmannGumbelQPolicy,
    BoltzmannQPolicy,
    EpsGreedyQPolicy,
    GreedyQPolicy,
    LinearAnnealedPolicy,
    MaxBoltzmannQPolicy,
)

# Project imports
path_root = Path(__file__).parents[2]
os.chdir(path_root)
sys.path.append(str(path_root))

import src.kniffel.classes.custom_exceptions as ex
from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.options import KniffelOptions
from src.kniffel_rl.env import KniffelEnv
from src.kniffel_rl.env_helper import KniffelEnvHelper
from src.kniffel_rl.env_helper import EnumAction
from src.utils.draw import KniffelDraw


class KniffelRL:
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

    # Current date
    datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")

    def __init__(
        self,
        load=False,
        test_episodes=100,
        path_prefix="",
        hyperparater_base={},
        config_path="src/ai/Kniffel.CSV",
        env_action_space=58,
        env_observation_space=20,
    ):
        self._load = load

        self._test_episodes = test_episodes
        self._config_path = config_path
        self._hyperparater_base = hyperparater_base

        self._env_action_space = env_action_space
        self._env_observation_space = env_observation_space

        if path_prefix == "":
            try:

                self._path_prefix = "/"
            except:
                self._path_prefix = ""
        else:
            self._path_prefix = path_prefix

    # Model
    def build_model(self, actions):
        model = tf.keras.Sequential()
        model.add(
            Flatten(
                input_shape=(
                    self.get_hyperparameter("windows_length"),
                    1,
                    self._env_observation_space,
                )
            )
        )

        for i in range(1, self.get_hyperparameter("layers") + 1):
            model.add(
                Dense(self.get_hyperparameter("n_units_l" + str(i)), activation="relu")
            )

        model.add(Dense(actions, activation=self.get_hyperparameter("activation")))
        # model.summary()
        return model

    def get_inner_policy(self):
        key = self.get_hyperparameter("linear_inner_policy")

        if key == "EpsGreedyQPolicy":
            policy = LinearAnnealedPolicy(
                EpsGreedyQPolicy(),
                attr="eps",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=1_000_000,
            )

        elif key == "BoltzmannQPolicy":

            policy = LinearAnnealedPolicy(
                BoltzmannQPolicy(),
                attr="tau",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=1_000_000,
            )

        if key == "MaxBoltzmannQPolicy":
            policy = LinearAnnealedPolicy(
                MaxBoltzmannQPolicy(),
                attr="eps",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=1_000_000,
            )

        return policy

    def get_policy(self, _key):
        key = self.get_hyperparameter(_key)

        policy = None
        if key == "LinearAnnealedPolicy":
            policy = self.get_inner_policy()

        elif key == "EpsGreedyQPolicy":

            policy = EpsGreedyQPolicy(eps=self.get_hyperparameter("eps_greedy_eps"))

        elif key == "GreedyQPolicy":

            policy = GreedyQPolicy()

        elif key == "BoltzmannQPolicy":

            policy = BoltzmannQPolicy(tau=self.get_hyperparameter("boltzmann_tau"))

        elif key == "MaxBoltzmannQPolicy":

            policy = MaxBoltzmannQPolicy(
                eps=self.get_hyperparameter("max_boltzmann_eps"),
                tau=self.get_hyperparameter("max_boltzmann_tau"),
            )
        elif key == "BoltzmannGumbelQPolicy":

            policy = BoltzmannGumbelQPolicy(
                C=self.get_hyperparameter("boltzmann_gumbel_C")
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

        a = self.get_hyperparameter("agent")

        if self.get_hyperparameter("agent") == "DQN":

            memory = SequentialMemory(
                limit=self.get_hyperparameter("dqn_memory_limit"),
                window_length=self.get_hyperparameter("windows_length"),
            )

            dqn_target_model_update = self.get_hyperparameter("dqn_target_model_update")

            enable_dueling_network = self.get_hyperparameter("enable_dueling_network")

            agent = DQNAgent(
                model=model,
                memory=memory,
                policy=self.get_policy("train_policy"),
                nb_actions=actions,
                nb_steps_warmup=self.get_hyperparameter("dqn_nb_steps_warmup"),
                enable_dueling_network=enable_dueling_network,
                target_model_update=int(round(dqn_target_model_update))
                if dqn_target_model_update > 0
                else float(dqn_target_model_update),
                batch_size=self.get_hyperparameter("batch_size"),
                enable_double_dqn=self.get_hyperparameter("dqn_enable_double_dqn"),
            )

            if enable_dueling_network:
                agent.dueling_type = self.get_hyperparameter("dqn_dueling_option")

        elif self.get_hyperparameter("agent") == "CEM":
            memory_interval = self.get_hyperparameter("cem_memory_limit")

            memory = EpisodeParameterMemory(
                limit=memory_interval,
                window_length=self.get_hyperparameter("windows_length"),
            )

            agent = CEMAgent(
                model=model,
                memory=memory,
                nb_actions=actions,
                nb_steps_warmup=self.get_hyperparameter("cem_nb_steps_warmup"),
                batch_size=self.get_hyperparameter("batch_size"),
                memory_interval=memory_interval,
            )

        elif self.get_hyperparameter("agent") == "SARSA":
            agent = SARSAAgent(
                model=model,
                policy=self.get_policy("train_policy"),
                test_policy=self.get_policy("test_policy"),
                nb_actions=actions,
                nb_steps_warmup=self.get_hyperparameter("sarsa_nb_steps_warmup"),
                delta_clip=self.get_hyperparameter("sarsa_delta_clip"),
                gamma=self.get_hyperparameter("sarsa_gamma"),
            )

        return agent

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

    def get_hyperparameter(self, key):
        return self._hyperparater_base[key]

    def train_agent(
        self,
        actions,
        env,
        nb_steps,
        callbacks,
        load_path="",
    ):
        model = self.build_model(actions)
        agent = self.build_agent(
            model,
            actions,
            nb_steps=nb_steps,
        )

        if (
            self.get_hyperparameter("agent") == "DQN"
            or self.get_hyperparameter("agent") == "SARSA"
        ):
            agent.compile(
                Adam(
                    learning_rate=self.get_hyperparameter(
                        "{}_adam_learning_rate".format(
                            self.get_hyperparameter("agent").lower()
                        )
                    ),
                    epsilon=self.get_hyperparameter(
                        "{}_adam_epsilon".format(
                            self.get_hyperparameter("agent").lower()
                        ),
                    ),
                ),
            )
        elif self.get_hyperparameter("agent") == "CEM":
            agent.compile()

        if self._load:
            print(f"Load existing model and train: path={load_path}/weights.h5f")
            agent.load_weights(f"{load_path}/weights.h5f")

        history = agent.fit(
            env,
            nb_steps=nb_steps,
            verbose=1,
            visualize=False,
            callbacks=callbacks,
            # action_repetition=2,
            log_interval=50_000,
        )

        return agent, history

    def validate_model(self, agent, env):
        scores = agent.test(env, nb_episodes=100, visualize=False)

        episode_reward = np.mean(scores.history["episode_reward"])
        nb_steps = np.mean(scores.history["nb_steps"])
        print(f"episode_reward: {episode_reward}")
        print(f"nb_steps: {nb_steps}")

        return scores

    def train(self, nb_steps=10_000, load_path="", env_config=""):
        return self._train(
            nb_steps=nb_steps, load_path=load_path, env_config=env_config
        )

    def _train(self, nb_steps=10_000, load_path="", env_config="", logging=False):
        env = KniffelEnv(
            env_config,
            config_file_path=self._config_path,
            env_observation_space=self._env_observation_space,
            env_action_space=self._env_action_space,
            logging=logging,
        )

        actions = env.action_space.n

        callbacks = []

        path = f"{self._path_prefix}output/weights/p_date={self.datetime}"

        # Create dir
        os.mkdir(path)

        print(f"Create subdir: {path}")

        # Create Callbacks
        checkpoint_weights_filename = path + "/weights_{step}.h5f"

        callbacks = [
            ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250_000)
        ]

        # Log
        log_file = path + "/log.json"

        callbacks += [FileLogger(log_file, interval=1_000)]

        callbacks += [EarlyStopping(patience=10, monitor="episode_reward", mode="max")]

        # Save configuration json
        json_object = json.dumps(hyperparameter, indent=4)

        self._append_file(f"{path}/configuration.json", json_object)

        agent, _ = self.train_agent(
            actions=actions,
            env=env,
            nb_steps=nb_steps,
            load_path=load_path,
            callbacks=callbacks,
        )

        test_scores = self.validate_model(agent, env=env)

        # save weights and configuration as json
        agent.save_weights(f"{path}/weights.h5f", overwrite=False)

        self.play(path, 10_000, env_config)

    def apply_prediction(
        self, kniffel: Kniffel, enum_action: EnumAction, logging=False
    ):
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

    def predict_and_apply(self, agent, kniffel: Kniffel, state, logging=False):
        action = agent.forward(state)

        enum_action = EnumAction(action)

        self.apply_prediction(kniffel, enum_action, logging)

        return enum_action

    # Use Model
    def play(
        self,
        path,
        episodes,
        env_config,
        random=False,
        logging=False,
        write=False,
        weights_name="weights",
    ):
        if random:
            print(f"Play {episodes} random games.")
            print()
            self.play_random(episodes, env_config)

        else:
            print(f"Play {episodes} games from model {path}.")
            print()
            self.use_model(
                path,
                episodes,
                env_config,
                weights_name=weights_name,
                logging=logging,
                write=write,
            )

    def play_random(self, episodes, env_config):
        env = KniffelEnv(
            env_config,
            logging=True,
            config_file_path=self._config_path,
            env_observation_space=self._env_observation_space,
            env_action_space=self._env_action_space,
        )

        bar = IncrementalBar("Games played", max=episodes)

        round = 1
        for episode in range(1, episodes + 1):
            bar.next()
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

        bar.finish()

    def build_use_agent(
        self,
        path,
        episodes,
        env_config,
        weights_name="weights",
        logging=False,
    ):
        env = KniffelEnv(
            env_config,
            logging=logging,
            config_file_path=self._config_path,
            env_observation_space=self._env_observation_space,
            env_action_space=self._env_action_space,
        )

        f = open(f"{path}/configuration.json")
        self._hyperparater_base = dict(json.load(f))

        actions = env.action_space.n

        model = self.build_model(actions)
        agent = self.build_agent(model, actions, nb_steps=episodes)
        if (
            self.get_hyperparameter("agent") == "DQN"
            or self.get_hyperparameter("agent") == "SARSA"
        ):
            agent.compile(
                Adam(
                    learning_rate=self.get_hyperparameter(
                        "{}_adam_learning_rate".format(
                            self.get_hyperparameter("agent").lower()
                        )
                    ),
                    epsilon=self.get_hyperparameter(
                        "{}_adam_epsilon".format(
                            self.get_hyperparameter("agent").lower()
                        ),
                    ),
                ),
            )
        elif self.get_hyperparameter("agent") == "CEM":
            agent.compile()

        if logging:
            print(f"Load weights {path}/{weights_name}.h5f")

        agent.load_weights(f"{path}/{weights_name}.h5f".format())

        return agent

    def use_model(
        self,
        path,
        episodes,
        env_config,
        weights_name="weights",
        logging=False,
        write=False,
    ):
        agent = self.build_use_agent(path, episodes, env_config, weights_name, logging)

        points = []
        rounds = []
        break_counter = 0

        bar = IncrementalBar(
            "Games played",
            max=episodes,
            suffix="%(index)d/%(max)d - %(eta)ds",
        )

        kniffel_env = KniffelEnvHelper(
            env_config=env_config,
            logging=logging,
            config_file_path=self._config_path,
        )

        for e in range(1, episodes + 1):
            bar.next()

            if write:
                self._append_file(
                    path=f"{path}/game_log/log.txt",
                    content="\n".join(KniffelDraw().draw_kniffel_title().split("\n")),
                )

            kniffel_env.reset_kniffel()

            rounds_counter = 1

            while True:
                log_csv = []

                state = kniffel_env.get_state()

                log_csv.append(
                    f"\n####################################################################################\n"
                )

                action = agent.forward(state)

                enum_action = EnumAction(action)

                log_csv.append(f"##  Try: {rounds_counter}\n")
                log_csv.append(f"##  Action: {enum_action}\n")

                log_csv.append("\n\n" + KniffelDraw().draw_dices(state[0][0:5]))

                # log_csv.append("\n" + KniffelDraw().draw_sheet(kniffel))

                reward, done, info = kniffel_env.predict_and_apply(action)
                agent.backward(reward, done)

                if not done:
                    rounds_counter += 1

                else:
                    agent.memory.append(state, action, 0, True, False)

                    if not info["error"]:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        log_csv.append(f"\nFinished/Error:")
                        log_csv.append(f"\n  Prediction Allowed: False")
                        log_csv.append(f"\n  Error: False")
                        log_csv.append(f"\n  Game Finished: True")

                        if write:
                            self._append_file(
                                path=f"{path}/game_log/log.txt",
                                content="\n" + ", ".join(log_csv),
                            )

                        break
                    else:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        log_csv.append("\n")
                        log_csv.append(f"Finished/Error:")
                        log_csv.append(f"\n  Error: {e}")
                        log_csv.append(f"\n  Prediction Allowed: False")
                        log_csv.append(f"\n  Error: True")

                        if write:
                            self._append_file(
                                path=f"{path}/game_log/log.txt",
                                content="\n" + "".join(log_csv),
                            )

                        break

                if write:
                    self._append_file(
                        path=f"{path}/game_log/log.txt",
                        content="\n" + "".join(log_csv),
                    )

                log_csv = []

            if write:
                self._append_file(path=f"{path}/game_log/log.txt", content="\n\n")

        bar.finish()

        print()
        print(f"Finished games: {episodes - break_counter}/{episodes}")
        print(f"Average points: {mean(points)}")
        print(f"Max points: {max(points)}")
        print(f"Min points: {min(points)}")
        print(f"Average rounds: {mean(rounds)}")
        print(f"Max rounds: {max(rounds)}")
        print(f"Min rounds: {min(rounds)}")

        return break_counter, sum(points) / len(points), max(points), min(points)


def play(rl: KniffelRL, env_config: dict):
    """Play a model

    Args:
        rl (KniffelRL): Kniffel RL Class
        env_config (dict): environment dict
    """
    rl.play(
        path="output/weights/p_date=2022-10-08-18_58_40",
        episodes=5000,
        env_config=env_config,
        weights_name="weights_250000",
        logging=False,
        write=False,
    )


def train(rl: KniffelRL, env_config: dict):
    """Train a model

    Args:
        rl (KniffelRL): Kniffel RL Class
        env_config (dict): environment dict
    """
    rl._train(
        nb_steps=20_000_000,
        env_config=env_config,
        load_path="output/weights/current-best-v2",
        logging=False,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    hyperparameter = {
        "agent": "DQN",
        "windows_length": 1,
        "layers": 3,
        "n_units_l1": 416,
        "n_units_l2": 496,
        "n_units_l3": 272,
        "activation": "linear",
        "dqn_memory_limit": 801000,
        "dqn_target_model_update": 4277.661580850757,
        "enable_dueling_network": True,
        "train_policy": "GreedyQPolicy",
        "dqn_nb_steps_warmup": 13,
        "batch_size": 256,
        "dqn_enable_double_dqn": False,
        "dqn_dueling_option": "max",
        "dqn_adam_learning_rate": 0.0014664720972469743,
        "dqn_adam_epsilon": 0.06434778285007711,
    }

    rl = KniffelRL(
        load=False,
        config_path="src/config/config.csv",
        path_prefix=str(Path(__file__).parents[2]) + "/",
        hyperparater_base=hyperparameter,
        env_observation_space=20,
        env_action_space=57,
    )

    env_config = {
        "reward_roll_dice": 0.5,
        "reward_game_over": -300,
        "reward_finish": 300,
        "reward_bonus": 50,
    }

    train(rl, env_config)
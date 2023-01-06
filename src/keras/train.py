# Standard imports
import numpy as np
import json
import os
import sys
import warnings
from datetime import datetime as dt
from pathlib import Path
from progress.bar import IncrementalBar
import gym

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Keras imports
import rl
from rl.agents import DQNAgent
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.memory import SequentialMemory
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
from src.env.open_ai_env import KniffelEnv
from src.env.env_helper import KniffelEnvHelper
from src.env.env_helper import EnumAction
from src.utils.draw import KniffelDraw


class KniffelRL:

    # OpenAI Gym environment
    env: KniffelEnv

    # Test episodes
    _test_episodes = 100

    # path prefix
    base_path = ""

    # Path to config csv
    _config_path = ""

    # Current date
    datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")

    # Env config
    env_config = {}

    # agent dict
    agent_dict = {}

    # logging
    logging = False

    # Env parameters
    _env_observation_space: int = 47
    _env_action_space: int = 57

    def __init__(
        self,
        agent_dict: dict,
        base_path: str = "",
        env_config={},
        config_path="src/ai/Kniffel.CSV",
        env_action_space=57,
        env_observation_space=47,
        logging=False,
    ):
        """Init the class

        :param agent_path: Path to agent json file, defaults to ""
        :param base_path: base path of project, defaults to ""
        :param env_config: env dict, defaults to {}
        :param config_path: path to config file, defaults to "src/ai/Kniffel.CSV"
        :param env_action_space: Action space, defaults to 57
        :param env_observation_space: Observation space, defaults to 47
        :param logging: use logging, defaults to False
        """

        self._config_path = config_path

        # Set env action space and observation space
        self._env_action_space = env_action_space
        self._env_observation_space = env_observation_space

        # Set env config
        self.env_config = env_config

        self.base_path = base_path

        self.logging = logging

        # if env_config has reward_mode
        if "reward_mode" in env_config:
            if (
                env_config["reward_mode"] == "kniffel"
                or env_config["reward_mode"] == "custom"
            ):
                print("Reward mode set to '{}'".format(env_config["reward_mode"]))
                self.reward_mode = env_config["reward_mode"]
            else:
                raise Exception(
                    "Reward mode {} is not supported. Please use 'kniffel' or 'custom'".format(
                        env_config["reward_mode"]
                    )
                )
        else:
            self.reward_mode = "kniffel"
            print("No reward mode set, using default 'kniffel'")

        # if env_config has state_mode
        if "state_mode" in env_config:
            if (
                env_config["state_mode"] == "binary"
                or env_config["state_mode"] == "continuous"
            ):
                print("State mode set to '{}'".format(env_config["state_mode"]))
                self.state_mode = env_config["state_mode"]
            else:
                raise Exception(
                    "Reward mode {} is not supported. Please use 'binary' or 'continuous'".format(
                        env_config["state_mode"]
                    )
                )
        else:
            self.state_mode = "binary"
            print("No state mode set, using default 'binary'")

        self.env = self.get_kniffel_env()

        self.agent_dict = agent_dict

    def get_hyperparameter(self, key):
        return self.agent_dict[key]

    # Model
    def build_model(self):
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
                Dense(
                    self.get_hyperparameter("n_units_l" + str(i)),
                    activation=self.get_hyperparameter("n_activation_l{}".format(i)),
                )
            )

        model.add(
            Dense(
                self._env_action_space, activation=self.get_hyperparameter("activation")
            )
        )
        model.summary()
        return model

    def get_inner_policy(self, anneal_steps) -> LinearAnnealedPolicy:
        key = self.get_hyperparameter("linear_inner_policy")

        # Set default to eps greedy q policy
        policy = LinearAnnealedPolicy(
            EpsGreedyQPolicy(),
            attr="eps",
            value_max=1,
            value_min=0.1,
            value_test=0.05,
            nb_steps=anneal_steps,
        )

        if key == "EpsGreedyQPolicy":
            policy = LinearAnnealedPolicy(
                EpsGreedyQPolicy(),
                attr="eps",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=anneal_steps,
            )

        elif key == "BoltzmannQPolicy":

            policy = LinearAnnealedPolicy(
                BoltzmannQPolicy(),
                attr="tau",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=anneal_steps,
            )

        elif key == "MaxBoltzmannQPolicy":
            policy = LinearAnnealedPolicy(
                MaxBoltzmannQPolicy(),
                attr="eps",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=anneal_steps,
            )

        return policy

    def get_policy(self, _key, anneal_steps):
        """
        Get policy

        :param _key: key of policy
        :param anneal_steps: number of steps
        :return: policy
        """
        key = self.get_hyperparameter(_key)

        # Use greedy q policy as default
        policy = GreedyQPolicy()

        if key == "LinearAnnealedPolicy":
            policy = self.get_inner_policy(anneal_steps)

        elif key == "EpsGreedyQPolicy":

            policy = EpsGreedyQPolicy(eps=self.get_hyperparameter("eps_greedy_eps"))

        elif key == "GreedyQPolicy":

            policy = GreedyQPolicy()

        elif key == "BoltzmannQPolicy":

            clip = self.get_hyperparameter("boltzmann_clip")

            policy = BoltzmannQPolicy(
                tau=self.get_hyperparameter("boltzmann_tau"), clip=(clip * -1, clip)
            )

        elif key == "MaxBoltzmannQPolicy":

            clip = self.get_hyperparameter("max_boltzmann_clip")

            policy = MaxBoltzmannQPolicy(
                eps=self.get_hyperparameter("max_boltzmann_eps"),
                tau=self.get_hyperparameter("max_boltzmann_tau"),
                clip=(clip * -1, clip),
            )
        elif key == "BoltzmannGumbelQPolicy":

            policy = BoltzmannGumbelQPolicy(
                C=self.get_hyperparameter("boltzmann_gumbel_C")
            )

        return policy

    def build_agent(self) -> DQNAgent:
        """Build the agent

        :param agent_dict: Agent dict
        :return: Agent
        """

        # Build neural network
        neural_network = self.build_model()

        # Build agent
        memory = SequentialMemory(
            limit=self.get_hyperparameter("dqn_memory_limit"),
            window_length=self.get_hyperparameter("windows_length"),
        )

        dqn_target_model_update = 0
        if "dqn_target_model_update_float" in self.agent_dict:
            dqn_target_model_update = self.get_hyperparameter(
                "dqn_target_model_update_float",
            )
        else:
            dqn_target_model_update = self.get_hyperparameter(
                "dqn_target_model_update_int",
            )

        enable_dueling_network = self.get_hyperparameter(
            "enable_dueling_network",
        )

        anneal_steps = self.get_hyperparameter("anneal_steps")

        # Agent
        agent = DQNAgent(
            model=neural_network,
            memory=memory,
            policy=self.get_policy("train_policy", anneal_steps),
            nb_actions=self._env_action_space,
            nb_steps_warmup=39,
            enable_dueling_network=bool(enable_dueling_network),
            target_model_update=int(round(dqn_target_model_update))
            if dqn_target_model_update > 0
            else float(dqn_target_model_update),
            batch_size=self.get_hyperparameter("batch_size"),
            enable_double_dqn=bool(self.get_hyperparameter("dqn_enable_double_dqn")),
            dueling_type="avg"
            if enable_dueling_network
            else str(self.get_hyperparameter("dqn_dueling_option")),
        )

        return agent

    def build_adam(self) -> Adam:
        """Build the adam optimizer

        :return: Optimizer
        """
        learning_rate = float(
            self.get_hyperparameter(
                "{}_adam_learning_rate".format(self.get_hyperparameter("agent").lower())
            )
        )

        beta_1 = float(
            self.get_hyperparameter(
                "{}_adam_beta_1".format(self.get_hyperparameter("agent").lower())
            )
        )

        beta_2 = float(
            self.get_hyperparameter(
                "{}_adam_beta_2".format(self.get_hyperparameter("agent").lower())
            )
        )

        epsilon = float(
            self.get_hyperparameter(
                "{}_adam_epsilon".format(self.get_hyperparameter("agent").lower()),
            )
        )

        amsgrad = bool(
            self.get_hyperparameter(
                "{}_adam_amsgrad".format(self.get_hyperparameter("agent").lower())
            )
        )

        optimizer = Adam(
            learning_rate=learning_rate,
            beta_1=beta_1,
            beta_2=beta_2,
            epsilon=epsilon,
            amsgrad=amsgrad,
        )

        return optimizer

    def get_kniffel_env(self) -> KniffelEnv:
        """Get the environment for the agent

        :return: Kniffel OpenAI environment
        """
        env = KniffelEnv(
            self.env_config,
            logging=self.logging,
            config_file_path=self._config_path,
            env_observation_space=self._env_observation_space,
            env_action_space=self._env_action_space,
            reward_mode=self.reward_mode,
            state_mode=self.state_mode,
        )

        return env

    def _append_file(self, path, content, retry=0):
        """Append content to file

        :param path: path of the file
        :param content: content to append
        :param retry: retry amount, defaults to 0
        """
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

    def validate_model(self, agent):
        scores = agent.test(self.env, nb_episodes=100, visualize=False)

        episode_reward = np.mean(scores.history["episode_reward"])
        nb_steps = np.mean(scores.history["nb_steps"])
        print(f"episode_reward: {episode_reward}")
        print(f"nb_steps: {nb_steps}")

    def train(
        self,
        load_dir_name="",
        nb_steps=10_000,
        load_weights=False,
    ):
        episodes = 1_000

        dir_name = "p_date={}/".format(self.datetime)

        path = os.path.join(self.base_path, "output/weights/", dir_name)

        # Create dir
        print(f"Create subdir: {path}")
        os.makedirs(path, exist_ok=True)

        # Build Agent
        agent = self.build_agent()

        # Get optimizer
        optimizer = self.build_adam()

        # compile the agent
        agent.compile(optimizer=optimizer, metrics=["mae"])

        # create callbacks
        callbacks = []

        dir_name = "p_date={}/".format(self.datetime)
        path = f"{self.base_path}output/weights/{dir_name}"

        # Create dir
        # os.mkdir(path)

        print(f"Create subdir: {path}")

        # Create Callbacks
        checkpoint_weights_filename = path + "/weights_{step}.h5f"

        callbacks = [
            ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250_000)
        ]

        # Log
        log_file = path + "/log.json"

        callbacks += [FileLogger(log_file, interval=1_000)]

        if load_weights:
            print(f"Load weights from dir: {load_dir_name}")
            agent.load_weights(
                f"{self.base_path}output/weights/{load_dir_name}/weights.h5f"
            )

        # fit the agent
        agent.fit(
            self.env,
            nb_steps=nb_steps,
            visualize=False,
            verbose=1,
            nb_max_episode_steps=39,
            callbacks=callbacks,
            log_interval=50_000,
        )

        # SAVE

        # 1. Save configuration json
        self._append_file(
            f"{path}/configuration.json", json.dumps(self.agent_dict, indent=4)
        )

        self._append_file(
            f"{path}/env_config.json",
            json.dumps(
                env_config,
                indent=4,
                sort_keys=True,
            ),
        )

        # 2. Save the weight from the model
        agent.save_weights(f"{path}/weights.h5f", overwrite=False)

        # TEST AND PLAY

        # Validate model
        self.validate_model(agent)

        # play
        metrics = self.play(dir_name, episodes)

        self._append_file(
            path=f"{path}/info.txt",
            content=json.dumps(metrics, indent=4, sort_keys=True),
        )

        return metrics

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

    # Play model
    def play(
        self,
        dir_name: str,
        episodes: int = 1_000,
        write=False,
    ):
        path = f"{self.base_path}output/weights/{dir_name}"

        agent_dict_play = {}
        # read json file to dict
        with open(f"{path}/configuration.json", "r") as f:
            agent_dict_play = json.load(f)

        self.agent_dict = agent_dict_play

        print(f"Play {episodes} games from model from path: {path}.")
        print()

        # Build Agent
        agent = self.build_agent()

        # Get optimizer
        optimizer = self.build_adam()

        # compile the agent
        agent.compile(optimizer=optimizer, metrics=["mae"])

        # load the weights
        agent.load_weights(f"{path}/weights.h5f")

        metrics = self.use_model(
            agent,
            path,
            episodes,
            write=write,
        )

        return metrics

    def calculate_custom_metric(self, l: list):
        sm_list = [np.power(v, 2) if v > 0 else -1 * np.power(v, 2) for v in l]  # type: ignore
        return np.mean(sm_list)

    def use_model(
        self,
        agent: DQNAgent,
        path: str,
        episodes: int,
        write: bool = False,
    ):
        points = []
        rounds = []
        break_counter = 0

        print("Start playing games...")

        bar = IncrementalBar(
            "Games played",
            max=episodes,
            suffix="%(index)d/%(max)d - %(eta)ds",
        )

        kniffel_env = KniffelEnvHelper(
            env_config=self.env_config,
            logging=self.logging,
            config_file_path=self._config_path,
        )

        for _ in range(1, episodes + 1):
            if write:
                self._append_file(
                    path=f"{path}/game_log/log.txt",
                    content="\n".join(KniffelDraw().draw_kniffel_title().split("\n")),
                )

            # reset values
            agent.reset_states()
            bar.next()
            kniffel_env.reset_kniffel()
            rounds_counter = 1
            done = False

            while not done:
                log_csv = []

                # Get fresh state
                state = kniffel_env.get_state()
                dices = kniffel_env.kniffel.get_last().get_latest().to_int_list()

                # predict action
                action = agent.forward(state)
                enum_action = EnumAction(action)

                # Apply action to model
                reward, done, info = kniffel_env.predict_and_apply(action)

                # Apply action to model
                agent.backward(reward, done)

                # DEBUG
                log_csv.append(
                    f"\n####################################################################################\n"
                )
                log_csv.append(f"##  Try: {rounds_counter}\n")
                log_csv.append(
                    f"##  Attempts left: {kniffel_env.kniffel.get_last().count()}/3\n"
                )
                log_csv.append(f"##  Action: {enum_action}\n")
                log_csv.append("\n\n" + KniffelDraw().draw_dices(state[0][0:30]))

                # log_csv.append("\n" + KniffelDraw().draw_sheet(kniffel_env.kniffel))

                if not done:
                    # if game not over increase round counter
                    rounds_counter += 1
                else:
                    if not info["error"]:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        log_csv.append(
                            "\n" + KniffelDraw().draw_sheet(kniffel_env.kniffel)
                        )

                        log_csv.append(
                            "\n\n####################################################################################\n"
                        )
                        log_csv.append(
                            "####################################################################################\n"
                        )
                        log_csv.append("##  Finished:\n")
                        log_csv.append("##    Error: False\n")
                        log_csv.append("##    Game Finished: True\n")

                        if write:
                            self._append_file(
                                path=f"{path}/game_log/log.txt",
                                content="\n\n" + "".join(log_csv),
                            )

                        break
                    else:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        log_csv.append(
                            "\n" + KniffelDraw().draw_sheet(kniffel_env.kniffel)
                        )

                        log_csv.append(
                            "\n\n####################################################################################"
                        )
                        log_csv.append(
                            "\n####################################################################################\n"
                        )
                        log_csv.append("##  Error:\n")
                        log_csv.append("##    Error: True\n")

                        error_description = str(info["exception_description"])
                        log_csv.append(f"##    Error: {error_description}\n")
                        log_csv.append("##    Prediction Allowed: False\n")

                        if write:
                            self._append_file(
                                path=f"{path}/game_log/log.txt",
                                content="\n\n" + "".join(log_csv),
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

        metrics = {
            "finished_games": episodes - break_counter,
            "error_games": break_counter,
            "rounds_played": episodes,
            "average_points": np.mean(points),
            "max_points": max(points),
            "min_points": min(points),
            "average_rounds": np.mean(rounds),
            "max_rounds": max(rounds),
            "min_rounds": min(rounds),
            "custom_metric": self.calculate_custom_metric(points),
        }

        print(json.dumps(metrics, indent=4, sort_keys=True))

        return metrics

    # Play model
    def evaluate(
        self,
        dir_name: str,
        episodes: int = 1_000,
    ):
        path = f"{self.base_path}output/weights/{dir_name}"

        print(f"Play {episodes} games from model from path: {path}.")
        print()

        # Build Agent
        agent = self.build_agent()

        # Get optimizer
        optimizer = self.build_adam()

        # compile the agent
        agent.compile(optimizer=optimizer, metrics=["mae"])

        # load the weights
        agent.load_weights(f"{path}/weights.h5f")

        self.evaluate_model(agent, episodes)

    def evaluate_model(
        self,
        agent: DQNAgent,
        episodes: int,
    ):
        points = []
        rounds = []
        break_counter = 0

        print("Start evaluating games...")

        bar = IncrementalBar(
            "Games played",
            max=episodes,
            suffix="%(index)d/%(max)d - %(eta)ds",
        )

        kniffel_env = KniffelEnvHelper(
            env_config=self.env_config,
            logging=self.logging,
            config_file_path=self._config_path,
        )

        action_dict = {}
        bonus_counter = 0  # count bonus

        for _ in range(1, episodes + 1):
            # reset values
            agent.reset_states()
            bar.next()
            kniffel_env.reset_kniffel()
            rounds_counter = 1
            done = False

            while not done:
                # Get fresh state
                state = kniffel_env.get_state()

                # predict action
                action = agent.forward(state)
                enum_action = EnumAction(action)

                # add enum action
                action_dict[enum_action] = (
                    action_dict[enum_action] + 1 if enum_action in action_dict else 1
                )

                # Apply action to model
                reward, done, info = kniffel_env.predict_and_apply(action)

                # Apply action to model
                agent.backward(reward, done)

                if not done:
                    # if game not over increase round counter
                    rounds_counter += 1
                else:
                    if not info["error"]:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        break
                    else:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        break

            bonus_counter += 1 if kniffel_env.kniffel.is_bonus() else 0

        bar.finish()

        for i in range(0, 56):
            if EnumAction(i) in action_dict:
                print(f"{EnumAction(i).name}: {action_dict[EnumAction(i)]}")
            else:
                print(f"{EnumAction(i).name}: 0")

        metrics = {
            "bonus_counter": bonus_counter,
            "finished_games": episodes - break_counter,
            "error_games": break_counter,
            "rounds_played": episodes,
            "average_points": np.mean(points),
            "max_points": max(points),
            "min_points": min(points),
            "average_rounds": np.mean(rounds),
            "max_rounds": max(rounds),
            "min_rounds": min(rounds),
        }

        print(json.dumps(metrics, indent=4, sort_keys=True))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    env_config = {
        "reward_roll_dice": 0.5,
        "reward_game_over": -25,
        "reward_finish": 25,
        "reward_bonus": 35,
        "reward_mode": "custom",
        "state_mode": "continuous",
    }

    agent_dict = {
        "activation": "linear",
        "agent": "DQN",
        "batch_size": 32,
        "dqn_adam_amsgrad": True,
        "dqn_adam_beta_1": 0.8770788026018081,
        "dqn_adam_beta_2": 0.8894717766504484,
        "dqn_adam_epsilon": 7.579405338028617e-05,
        "dqn_adam_learning_rate": 0.0015,
        "dqn_dueling_option": "avg",
        "dqn_enable_double_dqn": True,
        "dqn_memory_limit": 1500000,
        "dqn_target_model_update_int": 9954,
        "enable_dueling_network": False,
        "eps_greedy_eps": 0.22187387376395634,
        "layers": 3,
        "n_activation_l1": "relu",
        "n_activation_l2": "tanh",
        "n_activation_l3": "relu",
        "n_units_l1": 96,
        "n_units_l2": 128,
        "n_units_l3": 256,
        "train_policy": "EpsGreedyQPolicy",
        "windows_length": 3,
        "anneal_steps": 1000000,
    }

    rl = KniffelRL(
        agent_dict=agent_dict,
        base_path=str(Path(__file__).parents[2]) + "/",
        env_config=env_config,
        config_path="src/config/config.csv",
        env_observation_space=47,
        env_action_space=57,
    )

    rl.train(
        nb_steps=20_000_000,
        load_weights=False,
        load_dir_name="current-best-v3",
    )
    # rl.play(dir_name="current-best-v3", episodes=5, write=False)
    # rl.evaluate(dir_name="current-best-v2", episodes=10_000)

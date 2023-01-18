# Standard imports
import numpy as np
import json
import os
import sys
import warnings
from datetime import datetime as dt
from pathlib import Path
from progress.bar import IncrementalBar

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Tensorforce imports
from tensorforce import Runner, Agent, Environment
import tensorflow as tf

# Project imports
path_root = Path(__file__).parents[2]
os.chdir(path_root)
sys.path.append(str(path_root))

import src.kniffel.classes.custom_exceptions as ex
from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.options import KniffelOptions
from src.env.tensorforce_env import KniffelEnvTF
from src.env.env_helper import KniffelEnvHelper
from src.env.env_helper import EnumAction
from src.utils.draw import KniffelDraw


class KniffelRL:

    # OpenAI Gym environment
    environment: Environment = None  # type: ignore

    # Agent
    agent: Agent = None  # type: ignore

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

    # logging
    logging = False

    # Env parameters
    _env_observation_space: int = 47
    _env_action_space: int = 57

    def __init__(
        self,
        agent_path: str = "",
        base_path: str = "",
        env_config={},
        env_action_space=57,
        env_observation_space=47,
        logging=False,
    ):
        """Init the class

        :param agent_path: Path to agent json file, defaults to ""
        :param base_path: base path of project, defaults to ""
        :param env_config: env dict, defaults to {}
        :param env_action_space: Action space, defaults to 57
        :param env_observation_space: Observation space, defaults to 47
        :param logging: use logging, defaults to False
        """

        # Set env action space and observation space
        self._env_action_space = env_action_space
        self._env_observation_space = env_observation_space

        # Set env config
        self.env_config = env_config

        self.base_path = base_path

        # Create train dir and define path
        self.train_dir_name = "p_date={}/".format(self.datetime)
        self.train_path = os.path.join(
            self.base_path, "output/weights/", self.train_dir_name
        )

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

        # Create environment
        self.environment = self.get_kniffel_environment()

        if agent_path != "":
            self.agent = Agent.create(
                agent=agent_path,
                environment=self.environment,
                saver=dict(
                    directory=self.train_path, frequency=1000, max_checkpoints=5
                ),
                summarizer=dict(
                    directory=os.path.join(self.train_path, "_summaries"),
                    summaries="all",
                ),
            )  # type: ignore
        else:
            raise Exception("Agent not defined")

    def get_kniffel_environment(self) -> Environment:
        """Get the environment for the agent

        :return: OpenAI Gym environment
        """
        env = KniffelEnvTF(
            self.env_config,
            logging=self.logging,
            env_observation_space=self._env_observation_space,
            env_action_space=self._env_action_space,
            reward_mode=self.reward_mode,
            state_mode=self.state_mode,
        )

        environment = Environment.create(
            environment=env,
        )

        return environment

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

    def train(
        self,
        nb_steps=10_000,
    ):
        episodes = 1_000

        # Create dir
        print(f"Create subdir: {self.train_path}")
        os.makedirs(self.train_path, exist_ok=True)

        # Create runner
        runner = Runner(agent=self.agent, environment=self.environment)
        runner.run(
            num_episodes=nb_steps, save_best_agent=self.train_path, evaluation=True
        )

        # Save agent
        self.agent.save(directory=self.train_path, format="saved-model")

        # Close runner
        runner.close()

        # Get metrics
        metrics = self.play(self.train_dir_name, episodes)

        # Save metrics
        result = json.dumps(metrics, indent=4, sort_keys=True)
        self._append_file(path=f"{self.train_path}/info.txt", content=result)

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
        dir_name,
        episodes,
        write=False,
    ):
        print(f"Play {episodes} games from model {dir_name}.")
        print()

        metrics = self.use_model(
            dir_name,
            episodes,
            write=write,
        )

        return metrics

    def build_use_agent(
        self,
        path,
        env,
        weights_name="weights",
        logging=False,
    ):
        f = open(f"{path}/configuration.json")
        self._hyperparater_base = dict(json.load(f))

        if logging:
            print(f"Load weights {path}/{weights_name}.h5f")

        agent = Agent.load(directory=path, format="checkpoint", environment=env)

        return agent

    # Batch inputs
    def batch(self, x):
        return np.expand_dims(x, axis=0)

    # Unbatch outputs
    def unbatch(self, x):
        if isinstance(x, tf.Tensor):  # TF tensor to NumPy array
            x = x.numpy()  # type: ignore
        if x.shape == (1,):  # Singleton array to Python value
            return x.item()
        else:
            return np.squeeze(x, axis=0)

    def calculate_custom_metric(self, l: list):
        sm_list = [np.power(v, 2) if v > 0 else -1 * np.power(v, 2) for v in l]  # type: ignore
        return np.mean(sm_list)

    def use_model(
        self,
        dir_name,
        episodes,
        write=False,
    ):
        points = []
        rounds = []
        break_counter = 0

        print("Start playing games...")

        path = os.path.join("output", "weights", dir_name)

        agent_path = os.path.join(path, "best-model.json")

        print(f"Use agent from {agent_path}")

        bar = IncrementalBar(
            "Games played",
            max=episodes,
            suffix="%(index)d/%(max)d - %(eta)ds",
        )

        kniffel_env: KniffelEnvHelper = KniffelEnvHelper(
            self.env_config,
            logging=False,
            reward_mode=self.reward_mode,
            state_mode=self.state_mode,
        )

        agent: tf.keras.Model = tf.saved_model.load(export_dir=path)

        for e in range(1, episodes + 1):
            bar.next()

            if write:
                self._append_file(
                    path=f"{path}/game_log/log.txt",
                    content="\n".join(KniffelDraw().draw_kniffel_title().split("\n")),
                )

            kniffel_env.reset_kniffel()

            rounds_counter = 1

            done = False
            while not done:
                log_csv = []

                # Get current state
                state = kniffel_env.get_state()

                # Get action from state
                states = self.environment.reset()
                states = self.batch(states)
                auxiliaries = dict(mask=np.ones(shape=(1, 57), dtype=bool))
                deterministic = True

                # Get action
                action = agent.act(states, auxiliaries, deterministic)

                # Unbatch action
                actions = self.unbatch(action)

                # Convert action to EnumAction
                enum_action = EnumAction(actions)

                # Apply action
                reward, done, info = kniffel_env.predict_and_apply(action)

                # LOG

                log_csv.append(
                    f"\n####################################################################################\n"
                )

                log_csv.append(f"##  Try: {rounds_counter}\n")

                log_csv.append(
                    f"##  Attempts left: {kniffel_env.kniffel.get_last().attempts_left()}/3\n"
                )
                log_csv.append(f"##  Action: {enum_action}\n")

                log_csv.append("\n\n" + KniffelDraw().draw_dices(state[0][0:30]))

                log_csv.append("\n" + KniffelDraw().draw_sheet(kniffel_env.kniffel))

                if not done:
                    rounds_counter += 1

                else:
                    if not info["error"]:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        # LOG

                        log_csv.append(
                            "####################################################################################\n"
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
                                content="\n" + ", ".join(log_csv),
                            )

                        break
                    else:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        # LOG

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


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    env_config = {
        "reward_roll_dice": 0,
        "reward_game_over": -0,
        "reward_finish": 25,
        "reward_bonus": 100,
        "reward_mode": "custom",  # custom or kniffel
        "state_mode": "continuous",  # binary or continuous
        "reward_kniffel": {
            "reward_ones": {
                "reward_five_dices": 5,
                "reward_four_dices": 4.0,
                "reward_three_dices": 2.0,
                "reward_two_dices": -0,
                "reward_one_dice": -1,
                "reward_slash": -2,
            },
            "reward_twos": {
                "reward_five_dices": 10.0,
                "reward_four_dices": 8.0,
                "reward_three_dices": 6.0,
                "reward_two_dices": -2,
                "reward_one_dice": -3,
                "reward_slash": -4,
            },
            "reward_threes": {
                "reward_five_dices": 15.0,
                "reward_four_dices": 12.0,
                "reward_three_dices": 9.0,
                "reward_two_dices": -3,
                "reward_one_dice": -4.5,
                "reward_slash": -6,
            },
            "reward_fours": {
                "reward_five_dices": 20.0,
                "reward_four_dices": 16.0,
                "reward_three_dices": 12.0,
                "reward_two_dices": -4,
                "reward_one_dice": -6,
                "reward_slash": -8,
            },
            "reward_fives": {
                "reward_five_dices": 25.0,
                "reward_four_dices": 20.0,
                "reward_three_dices": 15.0,
                "reward_two_dices": -5,
                "reward_one_dice": -7.5,
                "reward_slash": -10,
            },
            "reward_sixes": {
                "reward_five_dices": 30.0,
                "reward_four_dices": 24.0,
                "reward_three_dices": 18.0,
                "reward_two_dices": -6,
                "reward_one_dice": -9,
                "reward_slash": -12,
            },
            "reward_three_times": {
                "reward_five_dices": 20.0,
                "reward_four_dices": 24.0,
                "reward_three_dices": 18.0,
                "reward_two_dices": 9.0,
                "reward_one_dice": 0.9,
                "reward_slash": -0,
            },
            "reward_four_times": {
                "reward_five_dices": 35.0,
                "reward_four_dices": 40.0,
                "reward_three_dices": 15.0,
                "reward_two_dices": 5,
                "reward_one_dice": 0.7,
                "reward_slash": -12,
            },
            "reward_full_house": {
                "reward_five_dices": 50.0,
                "reward_four_dices": None,
                "reward_three_dices": None,
                "reward_two_dices": None,
                "reward_one_dice": None,
                "reward_slash": -0,
            },
            "reward_small_street": {
                "reward_five_dices": 1.0,
                "reward_four_dices": 25.0,
                "reward_three_dices": None,
                "reward_two_dices": None,
                "reward_one_dice": None,
                "reward_slash": -0,
            },
            "reward_large_street": {
                "reward_five_dices": 60.0,
                "reward_four_dices": None,
                "reward_three_dices": None,
                "reward_two_dices": None,
                "reward_one_dice": None,
                "reward_slash": -0,
            },
            "reward_kniffel": {
                "reward_five_dices": 100.0,
                "reward_four_dices": None,
                "reward_three_dices": None,
                "reward_two_dices": None,
                "reward_one_dice": None,
                "reward_slash": -25,
            },
            "reward_chance": {
                "reward_five_dices": 5,
                "reward_four_dices": 4,
                "reward_three_dices": 3,
                "reward_two_dices": 2,
                "reward_one_dice": 1,
                "reward_slash": -0,
            },
        },
    }

    agent_path = "src/tensorforce/agent.json"

    rl = KniffelRL(
        agent_path=agent_path,
        base_path=str(Path(__file__).parents[2]) + "/",
        env_config=env_config,
        env_observation_space=47,
        env_action_space=57,
    )

    rl.train(10_000)
    # rl.play(dir_name="p_date=2022-12-19-22_11_30", episodes=5, write=True)

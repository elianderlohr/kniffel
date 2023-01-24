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

# SB3
from sb3_contrib import TRPO, ARS, QRDQN, RecurrentPPO
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Project imports
path_root = Path(__file__).parents[2]
os.chdir(path_root)
sys.path.append(str(path_root))

from src.env.sb3_env import KniffelEnvSB3
from src.env.open_ai_env import KniffelEnv

import src.kniffel.classes.custom_exceptions as ex
from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.options import KniffelOptions

from src.env.env_helper import KniffelEnvHelper
from src.env.env_helper import EnumAction
from src.utils.draw import KniffelDraw


class KniffelRL:

    # OpenAI Gym environment
    env: KniffelEnvSB3

    # Test episodes
    _test_episodes = 100

    # path prefix
    base_path = ""

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

    def get_kniffel_env(self) -> KniffelEnvSB3:
        """Get the environment for the agent

        :return: Kniffel OpenAI environment
        """
        env = KniffelEnvSB3(
            self.env_config,
            logging=self.logging,
            env_observation_space=self._env_observation_space,
            env_action_space=self._env_action_space,
            reward_mode=self.reward_mode,
            state_mode=self.state_mode,
        )

        from stable_baselines3.common.env_checker import check_env

        check_env(env, warn=True, skip_render_check=True)

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

    def build_sb_agent(self):
        if self.get_hyperparameter("agent") == "TRPO":
            prefix = "TRPO"
            return TRPO(
                policy=self.get_hyperparameter(f"{prefix}_policy"),
                env=self.get_kniffel_env(),
                batch_size=128,
                learning_rate=self.get_hyperparameter("learning_rate"),
                gamma=self.get_hyperparameter(f"{prefix}_gamma"),
                cg_max_steps=self.get_hyperparameter(f"{prefix}_cg_max_steps"),
                cg_damping=self.get_hyperparameter(f"{prefix}_cg_damping"),
                line_search_shrinking_factor=self.get_hyperparameter(
                    f"{prefix}_line_search_shrinking_factor"
                ),
                line_search_max_iter=self.get_hyperparameter(
                    f"{prefix}_line_search_max_iter",
                ),
                n_critic_updates=self.get_hyperparameter(
                    f"{prefix}_n_critic_updates",
                ),
                gae_lambda=self.get_hyperparameter(
                    f"{prefix}_gae_lambda",
                ),
                use_sde=False,
                normalize_advantage=self.get_hyperparameter(
                    f"{prefix}_normalize_advantage",
                ),
                target_kl=self.get_hyperparameter(
                    f"{prefix}_target_kl",
                ),
            )
        elif self.get_hyperparameter("agent") == "PPO":
            prefix = "PPO"
            return PPO(
                policy=self.get_hyperparameter(
                    f"{prefix}_policy",
                ),
                env=self.get_kniffel_env(),
                batch_size=128,
                learning_rate=self.get_hyperparameter(
                    f"{prefix}_learning_rate",
                ),
                gamma=self.get_hyperparameter(
                    f"{prefix}_gamma",
                ),
                gae_lambda=self.get_hyperparameter(
                    f"{prefix}_gae_lambda",
                ),
                clip_range=self.get_hyperparameter(
                    f"{prefix}_clip_range",
                ),
                normalize_advantage=self.get_hyperparameter(
                    f"{prefix}_normalize_advantage",
                ),
                ent_coef=self.get_hyperparameter(
                    f"{prefix}_ent_coef",
                ),
                vf_coef=self.get_hyperparameter(
                    f"{prefix}_vf_coef",
                ),
                max_grad_norm=self.get_hyperparameter(
                    f"{prefix}_max_grad_norm",
                ),
                use_sde=False,
                target_kl=self.get_hyperparameter(
                    f"{prefix}_target_kl",
                ),
            )
        elif self.get_hyperparameter("agent") == "DQN":
            prefix = "DQN"
            return DQN(
                policy=self.get_hyperparameter(
                    f"{prefix}_policy",
                ),
                env=self.get_kniffel_env(),
                learning_rate=0.003,
            )

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
        agent = self.build_sb_agent()

        # create callbacks
        callbacks = []

        dir_name = "p_date={}/".format(self.datetime)
        path = f"{self.base_path}output/weights/{dir_name}"

        if load_weights:
            print(f"Load weights from dir: {load_dir_name}")
            # agent.load_weights(
            #    f"{self.base_path}output/weights/{load_dir_name}/weights.h5f"
            # )

        # fit the agent
        agent.learn(total_timesteps=nb_steps, log_interval=10, progress_bar=True)  # type: ignore

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
        agent.save(f"{path}/kniffel_model")  # type: ignore

        # TEST AND PLAY

        # play
        metrics = self.evaluate(dir_name, episodes)

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
        agent = self.build_sb_agent()

        # load the weights
        agent.load(f"{path}/kniffel_model")  # type: ignore

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
        agent,
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

        env: KniffelEnvSB3 = self.get_kniffel_env()

        for _ in range(1, episodes + 1):
            if write:
                self._append_file(
                    path=f"{path}/game_log/log.txt",
                    content="\n".join(KniffelDraw().draw_kniffel_title().split("\n")),
                )

            state = env.reset()

            # reset values
            bar.next()
            rounds_counter = 1
            done = False

            while not done:
                log_csv = []

                # predict action
                action, _states = agent.predict(state, deterministic=True)
                enum_action = EnumAction(action)

                # Apply action to model
                obs, reward, done, info = env.step(action)

                # DEBUG
                log_csv.append(
                    f"\n####################################################################################\n"
                )
                log_csv.append(f"##  Try: {rounds_counter}\n")
                log_csv.append(
                    f"##  Attempts left: {env.kniffel_helper.kniffel.get_last().count()}/3\n"
                )
                log_csv.append(f"##  Action: {enum_action}\n")
                log_csv.append("\n\n" + KniffelDraw().draw_dices(state[0][0:30]))

                # log_csv.append("\n" + KniffelDraw().draw_sheet(kniffel_env.kniffel))

                if not done:
                    # if game not over increase round counter
                    rounds_counter += 1
                else:
                    if not info["error"]:
                        points.append(env.kniffel_helper.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        log_csv.append(
                            "\n" + KniffelDraw().draw_sheet(env.kniffel_helper.kniffel)
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
                        points.append(env.kniffel_helper.kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        log_csv.append(
                            "\n" + KniffelDraw().draw_sheet(env.kniffel_helper.kniffel)
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
        agent = self.build_sb_agent()

        # load the weights
        agent.load(f"{path}/kniffel_model", env=self.get_kniffel_env())  # type: ignore

        mean_reward, std_reward = evaluate_policy(
            agent, agent.get_env(), n_eval_episodes=100  # type: ignore
        )

        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

        return self.evaluate_model(agent, episodes, path=path)

    def evaluate_model(
        self,
        agent,
        episodes: int,
        path: str = "",
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

        env: KniffelEnvSB3 = self.get_kniffel_env()

        actions = []
        bonus_counter = 0  # count bonus

        for game_id in range(1, episodes + 1):

            # reset values
            bar.next()
            rounds_counter = 1
            done = False

            state = env.reset()

            while not done:
                # Get fresh state
                state = env.kniffel_helper.kniffel.get_state()

                # predict action
                action, _states = agent.predict(state, deterministic=True)
                enum_action = EnumAction(action)

                # Apply action to model

                obs, reward, done, info = env.step(action)

                if not done:
                    # if game not over increase round counter
                    rounds_counter += 1

                    # add action infos to list
                    action_dict = {
                        "game_id": game_id,
                        "round": rounds_counter,
                        "action": enum_action.name,
                        "points": env.kniffel_helper.kniffel.get_turn(-2)
                        .get_selected_option()
                        .points
                        if action >= 0 and action <= 12
                        else 0,
                    }

                    actions.append(action_dict)
                else:
                    if not info["error"]:
                        points.append(env.kniffel_helper.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        # add action infos to list
                        action_dict = {
                            "game_id": game_id,
                            "round": rounds_counter,
                            "action": enum_action.name,
                            "points": env.kniffel_helper.kniffel.get_turn(-1)
                            .get_selected_option()
                            .points
                            if action >= 0 and action <= 12
                            else 0,
                        }

                        actions.append(action_dict)

                        break
                    else:
                        points.append(env.kniffel_helper.kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        break

            bonus_counter += 1 if env.kniffel_helper.kniffel.is_bonus() else 0

        bar.finish()

        metrics = {
            "games": episodes,
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

        return_dict = metrics.copy()

        print(json.dumps(metrics, indent=4, sort_keys=True))

        # create evaluate dir in path if not exists
        if not os.path.exists(f"{path}/evaluate"):
            os.makedirs(f"{path}/evaluate")

        metrics["actions"] = actions

        with open(f"{path}/evaluate/metrics.json", "w") as final:
            json.dump(metrics, final, indent=4)

        return return_dict


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

    agent_dict = {"agent": "DQN", "DQN_policy": "MlpPolicy"}

    rl = KniffelRL(
        agent_dict=agent_dict,
        base_path=str(Path(__file__).parents[2]) + "/",
        env_config=env_config,
        env_observation_space=47,
        env_action_space=57,
    )

    TASK = "train"  # train, play, evaluate

    if TASK == "train":
        rl.train(
            nb_steps=10_000_000,
            load_weights=False,
            load_dir_name="current-best-v3",
        )
    elif TASK == "play":
        rl.play(dir_name="p_date=2023-01-24-09_02_29", episodes=1000, write=False)
    elif TASK == "evaluate":
        rl.evaluate(dir_name="p_date=2023-01-24-09_02_29", episodes=1000)

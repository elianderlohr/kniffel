# Standard imports
from datetime import datetime as dt
import warnings
from pathlib import Path
import sys
import multiprocessing
import argparse
import numpy as np
import optuna

# Keras imports

import tensorflow as tf

# Import SB3
from sb3_contrib import TRPO, ARS, QRDQN
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.evaluation import evaluate_policy

# Kniffel

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.env.sb3_env import KniffelEnvSB3
from src.env.env_helper import KniffelEnvHelper
from src.env.callback.custom_keras_pruning_callback import (
    CustomKerasPruningCallback,
)
from src.env.env_helper import EnumAction


class KniffelRL:
    """Optuna RL Kniffel Class"""

    # Hyperparameter object
    _hyperparater_base = {}

    # Test episodes
    _test_episodes = 100

    # Path to config csv
    _config_path = ""

    # Optuna trial
    _trial: optuna.trial.Trial = None  # type: ignore

    # Agent
    _agent_value: str = ""

    # Current date
    datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")

    def __init__(
        self,
        test_episodes=100,
        hyperparater_base={},
        trial: optuna.trial.Trial = None,  # type: ignore
        env_action_space=57,
        env_observation_space=20,
        env_config={},
    ):
        self.env_config = env_config

        self._hyperparater_base = hyperparater_base
        self._test_episodes = test_episodes
        self._trial = trial
        self._agent_value = self._return_trial("agent")

        self._env_action_space = env_action_space
        self._env_observation_space = env_observation_space

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

    def _return_trial(self, key):
        return self._trial.suggest_categorical(key, self._hyperparater_base[key])

    def build_agent(self, env):
        if self._agent_value == "TRPO":
            prefix = "TRPO"
            return TRPO(
                policy=self._trial.suggest_categorical(
                    f"{prefix}_policy", ["MlpPolicy"]
                ),
                env=env,
                batch_size=128,
                learning_rate=self._trial.suggest_float("learning_rate", 1e-5, 1e-1),
                gamma=self._trial.suggest_float(f"{prefix}_gamma", 0.9, 0.999),
                cg_max_steps=self._trial.suggest_int(f"{prefix}_cg_max_steps", 10, 100),
                cg_damping=self._trial.suggest_float(
                    f"{prefix}_cg_damping", 1e-3, 1e-1
                ),
                line_search_shrinking_factor=self._trial.suggest_float(
                    f"{prefix}_line_search_shrinking_factor", 0.1, 0.9
                ),
                line_search_max_iter=self._trial.suggest_int(
                    f"{prefix}_line_search_max_iter", 10, 100
                ),
                n_critic_updates=self._trial.suggest_int(
                    f"{prefix}_n_critic_updates", 10, 100
                ),
                gae_lambda=self._trial.suggest_float(
                    f"{prefix}_gae_lambda", 0.9, 0.999
                ),
                use_sde=False,
                normalize_advantage=self._trial.suggest_categorical(
                    f"{prefix}_normalize_advantage", [True, False]
                ),
                target_kl=self._trial.suggest_float(f"{prefix}_target_kl", 1e-3, 1e-1),
            )
        elif self._agent_value == "PPO":
            prefix = "PPO"
            return PPO(
                policy=self._trial.suggest_categorical(
                    f"{prefix}_policy", ["MlpPolicy"]
                ),
                env=env,
                batch_size=128,
                learning_rate=self._trial.suggest_float(
                    f"{prefix}_learning_rate", 1e-5, 1e-1
                ),
                gamma=self._trial.suggest_float(f"{prefix}_gamma", 0.9, 0.999),
                gae_lambda=self._trial.suggest_float(
                    f"{prefix}_gae_lambda", 0.9, 0.999
                ),
                clip_range=self._trial.suggest_float(f"{prefix}_clip_range", 0.1, 0.4),
                normalize_advantage=self._trial.suggest_categorical(
                    f"{prefix}_normalize_advantage", [True, False]
                ),
                ent_coef=self._trial.suggest_float(f"{prefix}_ent_coef", 1e-8, 1e-1),
                vf_coef=self._trial.suggest_float(f"{prefix}_vf_coef", 1e-8, 1e-1),
                max_grad_norm=self._trial.suggest_float(
                    f"{prefix}_max_grad_norm", 1e-1, 1e1
                ),
                use_sde=False,
                target_kl=self._trial.suggest_float(f"{prefix}_target_kl", 1e-3, 1e-1),
            )
        elif self._agent_value == "ARS":
            prefix = "ARS"
            return ARS(
                policy=self._trial.suggest_categorical(
                    f"{prefix}_policy", ["MlpPolicy", "LinearPolicy"]
                ),
                env=env,
                n_delta=self._trial.suggest_int(f"{prefix}_n_delta", 1, 100),
                learning_rate=self._trial.suggest_float(
                    f"{prefix}_learning_rate", 1e-5, 1e-1
                ),
                delta_std=self._trial.suggest_float(f"{prefix}_delta_std", 1e-5, 1e-1),
                n_top=self._trial.suggest_int(f"{prefix}_n_top", 1, 100),
                zero_policy=self._trial.suggest_categorical(
                    f"{prefix}_zero_policy", [True, False]
                ),
            )
        elif self._agent_value == "QRDQN":
            prefix = "QRDQN"
            return QRDQN(
                policy=self._trial.suggest_categorical(
                    f"{prefix}_policy", ["MlpPolicy"]
                ),
                env=env,
                batch_size=32,
                learning_rate=self._trial.suggest_float(
                    f"{prefix}_learning_rate", 1e-7, 1e-1
                ),
                buffer_size=self._trial.suggest_int(
                    f"{prefix}_buffer_size", 100_000, 2_000_000, step=100_000
                ),
                learning_starts=self._trial.suggest_int(
                    f"{prefix}_learning_starts", 100, 50_000, step=100
                ),
                tau=self._trial.suggest_float(f"{prefix}_tau", 0.1, 1.0),
                gamma=self._trial.suggest_float(f"{prefix}_gamma", 0.9, 0.999),
                train_freq=self._trial.suggest_int(f"{prefix}_train_freq", 1, 100),
                gradient_steps=self._trial.suggest_int(
                    f"{prefix}_gradient_steps", 1, 100
                ),
                target_update_interval=self._trial.suggest_int(
                    f"{prefix}_target_update_interval", 30, 20_000
                ),
                exploration_fraction=self._trial.suggest_float(
                    f"{prefix}_exploration_fraction", 0.001, 0.05
                ),
                exploration_final_eps=self._trial.suggest_float(
                    f"{prefix}_exploration_final_eps", 0.01, 1.0
                ),
                exploration_initial_eps=self._trial.suggest_float(
                    f"{prefix}_exploration_initial_eps", 0.01, 1.0
                ),
            )
        elif self._agent_value == "A2C":
            prefix = "A2C"
            return A2C(
                policy=self._trial.suggest_categorical(
                    f"{prefix}_policy", ["MlpPolicy"]
                ),
                env=env,
                learning_rate=self._trial.suggest_float(
                    f"{prefix}_learning_rate", 1e-7, 1e-1
                ),
                gamma=self._trial.suggest_float(f"{prefix}_gamma", 0.9, 0.999),
                gae_lambda=self._trial.suggest_float(f"{prefix}_gae_lambda", 0.9, 1),
                vf_coef=self._trial.suggest_float(f"{prefix}_vf_coef", 0.001, 0.5),
                max_grad_norm=self._trial.suggest_float(
                    f"{prefix}_max_grad_norm", 0.1, 1
                ),
                use_rms_prop=self._trial.suggest_categorical(
                    f"{prefix}_use_rms_prop", [True, False]
                ),
                use_sde=False,
            )
        elif self._agent_value == "DQN":
            prefix = "DQN"
            return DQN(
                policy=self._trial.suggest_categorical(
                    f"{prefix}_policy", ["MlpPolicy"]
                ),
                env=env,
                batch_size=32,
                learning_rate=self._trial.suggest_float(
                    f"{prefix}_learning_rate", 1e-7, 1e-1
                ),
                buffer_size=self._trial.suggest_int(
                    f"{prefix}_buffer_size", 100_000, 2_000_000, step=100_000
                ),
                learning_starts=self._trial.suggest_int(
                    f"{prefix}_learning_starts", 100, 50_000, step=100
                ),
                tau=self._trial.suggest_float(f"{prefix}_tau", 0.1, 1.0),
                gamma=self._trial.suggest_float(f"{prefix}_gamma", 0.9, 0.999),
                train_freq=self._trial.suggest_int(f"{prefix}_train_freq", 1, 100),
                gradient_steps=self._trial.suggest_int(
                    f"{prefix}_gradient_steps", 1, 100
                ),
                target_update_interval=self._trial.suggest_int(
                    f"{prefix}_target_update_interval", 30, 20_000
                ),
                exploration_fraction=self._trial.suggest_float(
                    f"{prefix}_exploration_fraction", 0.001, 0.05
                ),
                exploration_final_eps=self._trial.suggest_float(
                    f"{prefix}_exploration_final_eps", 0.01, 1.0
                ),
                exploration_initial_eps=self._trial.suggest_float(
                    f"{prefix}_exploration_initial_eps", 0.01, 1.0
                ),
                max_grad_norm=self._trial.suggest_float(
                    f"{prefix}_max_grad_norm", 0.1, 1
                ),
            )

    def train_agent(
        self,
        env,
        nb_steps,
    ):
        agent = self.build_agent(env)

        agent.learn(total_timesteps=nb_steps, log_interval=2)  # type: ignore

        return agent

    def calculate_custom_metric(self, l: list):
        sm_list = [np.power(v, 2) if v > 0 else -1 * np.power(v, 2) for v in l]
        return np.mean(sm_list)

    def train(self, nb_steps=10_000):
        env = KniffelEnvSB3(
            self.env_config,
            env_action_space=self._env_action_space,
            env_observation_space=self._env_observation_space,
            reward_mode=self.reward_mode,
            state_mode=self.state_mode,
        )

        agent = self.train_agent(
            env=env,
            nb_steps=nb_steps,
        )

        # Test agent
        metrics = self.test_model(agent=agent, env=env, episodes=self._test_episodes)

        # Return metrics
        return metrics

    def test_model(
        self,
        agent,
        env,
        episodes: int,
    ):
        points = []
        rounds = []
        break_counter = 0

        print("Start playing games...")

        for _ in range(1, episodes + 1):

            # reset values
            state = env.reset()
            rounds_counter = 1
            done = False

            while not done:
                # Get fresh state
                state = env.kniffel_helper.kniffel.get_state()
                # predict action
                action, _ = agent.predict(state, deterministic=True)

                # Apply action to model
                obs, reward, done, info = env.step(action)

                if not done:
                    # if game not over increase round counter
                    rounds_counter += 1
                else:
                    if not info["error"]:
                        points.append(env.kniffel_helper.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

                        break
                    else:
                        points.append(env.kniffel_helper.kniffel.get_points())
                        rounds.append(rounds_counter)
                        break_counter += 1
                        rounds_counter = 1

                        break

        # Get policy mean and std
        policy_mean_reward, policy_std_reward = evaluate_policy(
            agent, env, n_eval_episodes=episodes  # type: ignore
        )

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
            "policy_mean_reward": policy_mean_reward,
            "policy_std_reward": policy_std_reward,
        }

        return metrics


def objective(trial):

    # Define hyperparameters
    base_hp = {
        "agent": ["TRPO", "ARS", "QRDQN", "PPO", "A2C", "DQN"],
    }

    env_config = {
        "reward_roll_dice": 1,
        "reward_game_over": -25,
        "reward_finish": 25,
        "reward_bonus": 50,
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
                "reward_slash": -10,
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
                "reward_slash": -12,
            },
            "reward_small_street": {
                "reward_five_dices": 1.0,
                "reward_four_dices": 25.0,
                "reward_three_dices": None,
                "reward_two_dices": None,
                "reward_one_dice": None,
                "reward_slash": -5,
            },
            "reward_large_street": {
                "reward_five_dices": 60.0,
                "reward_four_dices": None,
                "reward_three_dices": None,
                "reward_two_dices": None,
                "reward_one_dice": None,
                "reward_slash": -15,
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
                "reward_slash": -1,
            },
        },
    }

    rl = KniffelRL(
        test_episodes=100,
        hyperparater_base=base_hp,
        trial=trial,
        env_config=env_config,
        env_observation_space=47,
        env_action_space=57,
    )

    metrics = rl.train(nb_steps=1_000_000)  # todo

    trial.set_user_attr("server", str(server))
    trial.set_user_attr("param", trial.params)

    trial.set_user_attr("max_points", float(metrics["max_points"]))
    trial.set_user_attr("min_points", float(metrics["min_points"]))
    trial.set_user_attr("average_points", float(metrics["average_points"]))

    trial.set_user_attr("max_rounds", float(metrics["max_rounds"]))
    trial.set_user_attr("min_rounds", float(metrics["min_rounds"]))
    trial.set_user_attr("average_rounds", float(metrics["average_rounds"]))

    trial.set_user_attr("policy_mean_reward", float(metrics["policy_mean_reward"]))
    trial.set_user_attr("policy_std_reward", float(metrics["policy_std_reward"]))

    return float(metrics["max_points"])


server = ""

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    cpu_count = multiprocessing.cpu_count()

    parser = argparse.ArgumentParser()
    parser.add_argument("--pw", type=str, default=None)
    parser.add_argument("--study_name", type=str, default="test")
    parser.add_argument("--new", choices=["true", "false"], default="false", type=str)
    parser.add_argument("--jobs", type=int, default=cpu_count)
    parser.add_argument("--server", type=str, default="test")

    print()

    args = parser.parse_args()

    server = args.server

    if args.new == "true":
        print(
            f"Create new study with name '{args.study_name}' and {args.jobs} parallel jobs. Run on server {args.server}"
        )
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            storage=f"mysql+pymysql://kniffeluser:{args.pw}@kniffel.mysql.database.azure.com:3306/optuna3_new",
        )
    else:
        print(
            f"Load study with name '{args.study_name}' and {args.jobs} parallel jobs. Run on server {args.server}"
        )
        study = optuna.load_study(
            study_name=args.study_name,
            storage=f"mysql+pymysql://kniffeluser:{args.pw}@kniffel.mysql.database.azure.com:3306/optuna3_new",
        )

    study.optimize(
        objective,
        n_trials=1000,
        catch=(ValueError,),
        n_jobs=args.jobs,
    )

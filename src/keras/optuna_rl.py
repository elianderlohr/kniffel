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

from rl.agents import DQNAgent, CEMAgent, SARSAAgent
from rl.policy import (
    BoltzmannQPolicy,
    EpsGreedyQPolicy,
    LinearAnnealedPolicy,
    GreedyQPolicy,
    MaxBoltzmannQPolicy,
    BoltzmannGumbelQPolicy,
)

from rl.memory import SequentialMemory, EpisodeParameterMemory

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

# Kniffel

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.env.open_ai_env import KniffelEnv
from src.env.callback.custom_keras_pruning_callback import (
    CustomKerasPruningCallback,
)


class KniffelRL:
    """Optuna RL Kniffel Class"""

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

    # Agent
    _agent_value: str = ""

    # Current date
    datetime = dt.today().strftime("%Y-%m-%d-%H_%M_%S")

    # Window
    window_length = 1

    def __init__(
        self,
        test_episodes=100,
        path_prefix="",
        hyperparater_base={},
        config_path="src/config/config.csv",
        trial: optuna.trial.Trial = None,
        env_action_space=57,
        env_observation_space=20,
        env_config={},
    ):
        self.env_config = env_config

        self._hyperparater_base = hyperparater_base
        self._test_episodes = test_episodes
        self._config_path = config_path
        self._trial = trial
        self._agent_value = self._return_trial("agent")

        # Hardcode window_length to 1 if agent is not DQN to reduce failed runs
        self.window_length = (
            self._return_trial("windows_length") if self._agent_value == "DQN" else 1
        )

        self._env_action_space = env_action_space
        self._env_observation_space = env_observation_space

        if path_prefix == "":
            try:
                import google.colab

                self._path_prefix = "/"
            except:
                self._path_prefix = ""
        else:
            self._path_prefix = path_prefix

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

    # Model
    def build_model(self, actions):
        model = tf.keras.Sequential()
        model.add(
            Flatten(
                input_shape=(
                    self.window_length,
                    1,
                    self._env_observation_space,
                )
            )
        )

        layers = self._trial.suggest_int("layers", 2, 4)

        for i in range(1, layers + 1):
            model.add(
                Dense(
                    self._trial.suggest_int("n_units_l{}".format(i), 32, 256, step=32),
                    activation=self._trial.suggest_categorical(
                        "n_activation_l{}".format(i), ["relu", "tanh"]
                    ),
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

    def get_inner_policy(self):
        key = self._return_trial("linear_inner_policy")

        if key == "EpsGreedyQPolicy":
            policy = LinearAnnealedPolicy(
                EpsGreedyQPolicy(),
                attr="eps",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=250_000,
            )

        elif key == "BoltzmannQPolicy":

            policy = LinearAnnealedPolicy(
                BoltzmannQPolicy(),
                attr="tau",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=250_000,
            )

        if key == "MaxBoltzmannQPolicy":
            policy = LinearAnnealedPolicy(
                MaxBoltzmannQPolicy(),
                attr="eps",
                value_max=1,
                value_min=0.1,
                value_test=0.05,
                nb_steps=250_000,
            )

        return policy

    def get_policy(self, key):

        policy = None
        if key == "LinearAnnealedPolicy":
            policy = self.get_inner_policy()

        elif key == "EpsGreedyQPolicy":

            policy = EpsGreedyQPolicy(
                eps=self._trial.suggest_float("eps_greedy_eps", 1e-5, 1)
            )

        elif key == "GreedyQPolicy":

            policy = GreedyQPolicy()

        elif key == "BoltzmannQPolicy":

            clip = self._trial.suggest_int("boltzmann_clip", 200, 1000, step=100)

            policy = BoltzmannQPolicy(
                tau=self._trial.suggest_float("boltzmann_tau", 0.05, 1, step=0.05),
                clip=(clip * -1, clip),
            )

        elif key == "MaxBoltzmannQPolicy":

            clip = self._trial.suggest_int("max_boltzmann_clip", 200, 1000, step=100)

            policy = MaxBoltzmannQPolicy(
                eps=self._trial.suggest_float("max_boltzmann_eps", 1e-5, 1),
                tau=self._trial.suggest_float("max_boltzmann_tau", 0.05, 1, step=0.05),
                clip=(clip * -1, clip),
            )
        elif key == "BoltzmannGumbelQPolicy":

            policy = BoltzmannGumbelQPolicy(
                C=self._trial.suggest_float("boltzmann_gumbel_C", 1e-5, 1)
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

        if self._agent_value == "DQN":
            memory = SequentialMemory(
                limit=self._trial.suggest_int(
                    "dqn_memory_limit", 50_000, 2_000_000, step=50_000
                ),
                window_length=self.window_length,
            )

            # dqn_target_model_update_float = self._trial.suggest_categorical(
            #    "dqn_target_model_update_is_float", [True, False]
            # )

            dqn_target_model_update_float = False

            dqn_target_model_update = 0
            if dqn_target_model_update_float:
                dqn_target_model_update = self._trial.suggest_float(
                    "dqn_target_model_update_float", 1e-6, 1e-1
                )
            else:
                dqn_target_model_update = self._trial.suggest_int(
                    "dqn_target_model_update_int", 1, 10_000
                )

            enable_dueling_network = self._trial.suggest_categorical(
                "enable_dueling_network", [False]
            )

            agent = DQNAgent(
                model=model,
                memory=memory,
                policy=self.get_policy(self._return_trial("train_policy")),
                nb_actions=actions,
                nb_steps_warmup=39,
                enable_dueling_network=bool(enable_dueling_network),
                target_model_update=int(round(dqn_target_model_update))
                if dqn_target_model_update > 0
                else float(dqn_target_model_update),
                batch_size=self._return_trial("batch_size"),
                enable_double_dqn=bool(self._return_trial("dqn_enable_double_dqn")),
                dueling_type="avg"
                if enable_dueling_network
                else str(self._return_trial("dqn_dueling_option")),
            )

        elif self._agent_value == "CEM":
            memory_interval = self._trial.suggest_int(
                "cem_memory_limit", 1_000, 750_000, step=50_000
            )

            memory = EpisodeParameterMemory(
                limit=memory_interval,
                window_length=self.window_length,
            )

            agent = CEMAgent(
                model=model,
                memory=memory,
                nb_actions=actions,
                nb_steps_warmup=25,
                batch_size=self._return_trial("batch_size"),
                memory_interval=memory_interval,
            )

        elif self._agent_value == "SARSA":
            agent = SARSAAgent(
                model=model,
                policy=self.get_policy(self._return_trial("train_policy")),
                test_policy=self.get_policy(self._return_trial("test_policy")),
                nb_actions=actions,
                nb_steps_warmup=25,
                delta_clip=self._trial.suggest_float("sarsa_delta_clip", 0.01, 0.99),
                gamma=self._trial.suggest_float("sarsa_gamma", 0.01, 0.99),
            )

        return agent

    def train_agent(
        self,
        actions,
        env,
        nb_steps,
    ):
        model = self.build_model(actions)

        agent = self.build_agent(model, actions, nb_steps=nb_steps)

        if self._agent_value == "DQN" or self._agent_value == "SARSA":
            _learning_rate = self._trial.suggest_float(
                "{}_adam_learning_rate".format(self._agent_value.lower()), 1e-6, 1e-2
            )

            # _beta1 = self._trial.suggest_float(
            #    "{}_adam_beta_1".format(self._agent_value.lower()), 0.6, 1
            # )
            # _beta2 = self._trial.suggest_float(
            #    "{}_adam_beta_2".format(self._agent_value.lower()), 0.6, 1
            # )
            # _epsilon = self._trial.suggest_float(
            #    "{}_adam_epsilon".format(self._agent_value.lower()), 1e-8, 1e-4
            # )
            # _amsgrad = self._trial.suggest_categorical(
            #    "{}_adam_amsgrad".format(self._agent_value.lower()), [False, True]
            # )
            _beta1 = 0.8770788026018081
            _beta2 = 0.8894717766504484
            _epsilon = 7.579405338028617e-05
            _amsgrad = True

            agent.compile(
                Adam(
                    learning_rate=_learning_rate,
                    beta_1=_beta1,
                    beta_2=_beta2,
                    epsilon=_epsilon,
                    amsgrad=bool(_amsgrad),
                ),
            )
        elif self._agent_value == "CEM":
            agent.compile()

        callbacks = []
        callbacks += [
            CustomKerasPruningCallback(self._trial, "episode_reward", interval=20_000),
        ]

        history = agent.fit(
            env,
            nb_steps=nb_steps,
            verbose=1,
            visualize=False,
            # action_repetition=2,
            log_interval=250_000,
            callbacks=callbacks,
        )

        return agent, history

    def calculate_custom_metric(self, l: list):
        sm_list = [np.power(v, 2) if v > 0 else -1 * np.power(v, 2) for v in l]
        return np.mean(sm_list)

    def validate_model(self, agent, env):
        scores = agent.test(env, nb_episodes=100, visualize=False)

        episode_reward_max = float(np.max(scores.history["episode_reward"]))
        episode_reward_min = float(np.min(scores.history["episode_reward"]))
        episode_reward_mean = float(np.mean(scores.history["episode_reward"]))

        episode_reward_custom = float(
            self.calculate_custom_metric(scores.history["episode_reward"])
        )

        nb_steps_max = float(np.max(scores.history["nb_steps"]))
        nb_steps_min = float(np.min(scores.history["nb_steps"]))
        nb_steps_mean = float(np.mean(scores.history["nb_steps"]))

        nb_steps_custom = float(
            self.calculate_custom_metric(scores.history["nb_steps"])
        )

        custom_metric = float(episode_reward_custom + (nb_steps_custom * 10))

        print(f"episode_reward_custom: {episode_reward_custom}")
        print(f"nb_steps_custom: {nb_steps_custom}")
        print(f"custom_metric: {custom_metric}")

        return (
            scores.history["episode_reward"],
            episode_reward_max,
            episode_reward_min,
            episode_reward_mean,
            episode_reward_custom,
            scores.history["nb_steps"],
            nb_steps_max,
            nb_steps_min,
            nb_steps_mean,
            nb_steps_custom,
            custom_metric,
        )

    def train(self, nb_steps=10_000):
        env = KniffelEnv(
            self.env_config,
            config_file_path=self._config_path,
            env_action_space=self._env_action_space,
            env_observation_space=self._env_observation_space,
            reward_mode=self.reward_mode,
            state_mode=self.state_mode,
        )

        agent, _ = self.train_agent(
            actions=self._env_action_space,
            env=env,
            nb_steps=nb_steps,
        )

        (
            episode_reward,
            episode_reward_max,
            episode_reward_min,
            episode_reward_mean,
            episode_reward_custom,
            nb_steps,
            nb_steps_max,
            nb_steps_min,
            nb_steps_mean,
            nb_steps_custom,
            custom_metric,
        ) = self.validate_model(agent, env=env)

        return (
            episode_reward,
            episode_reward_max,
            episode_reward_min,
            episode_reward_mean,
            episode_reward_custom,
            nb_steps,
            nb_steps_max,
            nb_steps_min,
            nb_steps_mean,
            nb_steps_custom,
            custom_metric,
        )


def objective(trial):

    base_hp = {
        "windows_length": [1, 2, 3, 4, 5],
        "batch_size": [32],
        "dqn_dueling_option": ["avg", "max"],
        "activation": ["linear"],
        "dqn_enable_double_dqn": [True],
        "agent": ["DQN"],
        "train_policy": ["EpsGreedyQPolicy"],
    }

    env_config = {
        "reward_roll_dice": 0.5,
        "reward_game_over": -25,
        "reward_finish": 25,
        "reward_bonus": 35,
        "reward_mode": "custom",
        "state_mode": "continuous",
    }

    rl = KniffelRL(
        hyperparater_base=base_hp,
        config_path="src/config/config.csv",
        path_prefix="",
        trial=trial,
        env_observation_space=47,
        env_action_space=57,
        env_config=env_config,
    )

    (
        episode_reward,
        episode_reward_max,
        episode_reward_min,
        episode_reward_mean,
        episode_reward_custom,
        nb_steps,
        nb_steps_max,
        nb_steps_min,
        nb_steps_mean,
        nb_steps_custom,
        custom_metric,
    ) = rl.train(nb_steps=1_000_000)

    trial.set_user_attr("server", str(server))
    # trial.set_user_attr("custom_metric", float(custom_metric))
    # trial.set_user_attr("episode_reward_custom", float(episode_reward_custom))
    # trial.set_user_attr("nb_steps_custom", float(nb_steps_custom))
    trial.set_user_attr("param", trial.params)

    # trial.set_user_attr("episode_reward", list(episode_reward))
    trial.set_user_attr("episode_reward_max", float(episode_reward_max))
    trial.set_user_attr("episode_reward_min", float(episode_reward_min))
    trial.set_user_attr("episode_reward_mean", float(episode_reward_mean))

    # trial.set_user_attr("nb_steps", list(nb_steps))
    trial.set_user_attr("nb_steps_max", float(nb_steps_max))
    trial.set_user_attr("nb_steps_min", float(nb_steps_min))
    trial.set_user_attr("nb_steps_mean", float(nb_steps_mean))

    return float(episode_reward_custom)


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

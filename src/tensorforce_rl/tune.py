# Standard imports
import numpy as np
import json
import os
import sys
import warnings
from datetime import datetime as dt
from pathlib import Path
from progress.bar import IncrementalBar
import math
import pickle

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Tuner import
import ConfigSpace as cs
from hpbandster.core.nameserver import NameServer, nic_name_to_host
from hpbandster.core.result import json_result_logger, logged_results_to_HBS_result
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

# Stable Baselines imports
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

class KniffelOptunaRL:

    # OpenAI Gym environment
    environment : Environment = None # type: ignore

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

    # optuna trial
    trial = None

    # Env parameters
    _env_observation_space: int = 47
    _env_action_space: int = 57

    def __init__(
        self,#
        trial: optuna.trial.Trial,
        agent,
        environment,
        base_path:str="",
        env_config={},
        config_path="src/ai/Kniffel.CSV",
        env_action_space=57,
        env_observation_space=47,
        logging=False,
    ):
        """ Init the class

        :param trial: optuna trial
        :param agent: Agent, defaults to None
        :param agent_path: Path to agent json file, defaults to ""
        :param base_path: base path of project, defaults to ""
        :param env_config: env dict, defaults to {}
        :param config_path: path to config file, defaults to "src/ai/Kniffel.CSV"
        :param env_action_space: Action space, defaults to 57
        :param env_observation_space: Observation space, defaults to 47
        :param logging: use logging, defaults to False
        """

        self.trial = trial

        self._config_path = config_path

        # Set env action space and observation space
        self._env_action_space = env_action_space
        self._env_observation_space = env_observation_space

        # Set env config
        self.env_config = env_config

        self.base_path = base_path            

        self.logging = logging

        self.environment = environment

        if agent is not None:
            self.agent = agent
        else:
            raise Exception("Agent not defined")

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

        reward_simple = self.env_config["reward_simple"]

        dir_name = "p_date={}/".format(self.datetime)

        path = os.path.join(
            self.base_path, "output/weights/", dir_name
        )

        # Create dir
        print(f"Create subdir: {path}")
        os.makedirs(path, exist_ok=True)

        if reward_simple:
            print("Use simple reward system!")
        else:
            print("Use complex reward system!")

        runner = Runner(agent=self.agent, environment=self.environment)

        runner.run(num_episodes=nb_steps, save_best_agent=path, evaluation=True)

        self.agent.save(directory=path, format="saved-model")

        runner.close()

        metrics = self.play(dir_name, episodes)

        result = json.dumps(metrics, indent=4, sort_keys=True)

        self._append_file(path=f"{path}/info.txt", content=result)

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

        metrics  = self.use_model(
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

        agent = Agent.load(directory=path, format='checkpoint', environment=env)

        return agent

    # Batch inputs
    def batch(self, x):
        return np.expand_dims(x, axis=0)

    # Unbatch outputs
    def unbatch(self, x):
        if isinstance(x, tf.Tensor):  # TF tensor to NumPy array
            x = x.numpy()
        if x.shape == (1,):  # Singleton array to Python value
            return x.item()
        else:
            return np.squeeze(x, axis=0)

    def calculate_custom_metric(self, l: list):
        """Calculate second moment with negative value in account

        :param l: list of values
        :return: second moment with negative values
        """
        sm_list = [np.power(v, 2) if v > 0 else -1 * np.power(v, 2) for v in l]
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

        path = os.path.join("output", "weights", dir_name)

        agent_path = os.path.join(path, "best-model.json")

        print(f"Use agent from {agent_path}")

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

        agent = tf.saved_model.load(export_dir=path)

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

                state = kniffel_env.get_state()

                states = self.environment.reset()
                states = self.batch(states)
                auxiliaries = dict(mask=np.ones(shape=(1, 57), dtype=bool))
                deterministic = True

                action = agent.act(states, auxiliaries, deterministic)

                log_csv.append(
                    f"\n####################################################################################\n"
                )

                actions = self.unbatch(action)

                enum_action = EnumAction(actions)

                log_csv.append(f"##  Try: {rounds_counter}\n")
                log_csv.append(
                    f"##  Attempts left: {kniffel_env.kniffel.get_last().attempts_left()}/3\n"
                )
                log_csv.append(f"##  Action: {enum_action}\n")

                log_csv.append("\n\n" + KniffelDraw().draw_dices(state[0][0:30]))

                reward, done, info = kniffel_env.predict_and_apply(action)

                log_csv.append("\n" + KniffelDraw().draw_sheet(kniffel_env.kniffel))

                if not done:
                    rounds_counter += 1

                else:
                    if not info["error"]:
                        points.append(kniffel_env.kniffel.get_points())
                        rounds.append(rounds_counter)
                        rounds_counter = 1

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

class TensorforceWorker(Worker):

    def __init__(
        self, *args, environment, num_episodes, base, runs_per_round, max_episode_timesteps=None,
        num_parallel=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.environment = environment
        self.max_episode_timesteps = max_episode_timesteps
        self.num_episodes = num_episodes
        self.base = base
        self.runs_per_round = runs_per_round
        self.num_parallel = num_parallel

    def compute(self, config_id, config, budget, working_directory):
        budget = math.log(budget, self.base)
        assert abs(budget - round(budget)) < util.epsilon
        budget = round(budget)
        assert budget < len(self.runs_per_round)
        num_runs = self.runs_per_round[budget]

        update = dict(unit='episodes', batch_size=config['batch_size'], frequency=1)
        policy = dict(network=dict(type='auto', size=64, depth=2, rnn=False))
        optimizer = dict(
            optimizer='adam', learning_rate=config['learning_rate'],
            multi_step=config['multi_step'], linesearch_iterations=5  # , subsampling_fraction=256
        )

        if config['clipping_value'] > 1.0:
            objective = dict(
                type='policy_gradient',
                importance_sampling=(config['importance_sampling'] == 'yes')
            )
        else:
            objective = dict(
                type='policy_gradient',
                importance_sampling=(config['importance_sampling'] == 'yes'),
                clipping_value=config['clipping_value']
            )

        if config['baseline'] == 'no':
            predict_horizon_values = False
            estimate_advantage = False
            predict_action_values = False
            baseline = None
            baseline_optimizer = None
            baseline_objective = None

        elif config['baseline'] == 'same':
            predict_horizon_values = 'early'
            estimate_advantage = (config['estimate_advantage'] == 'yes')
            predict_action_values = False
            baseline = None
            baseline_optimizer = config['baseline_weight']
            baseline_objective = dict(type='value', value='state')

        elif config['baseline'] == 'yes':
            predict_horizon_values = 'early'
            estimate_advantage = (config['estimate_advantage'] == 'yes')
            predict_action_values = False
            baseline = dict(network=dict(type='auto', size=64, depth=2, rnn=False))
            baseline_optimizer = config['baseline_weight']
            baseline_objective = dict(type='value', value='state')

        else:
            assert False

        reward_estimation = dict(
            horizon=config['horizon'], discount=config['discount'],
            predict_horizon_values=predict_horizon_values, estimate_advantage=estimate_advantage,
            predict_action_values=predict_action_values
        )

        if config['entropy_regularization'] < 1e-5:
            entropy_regularization = 0.0
        else:
            entropy_regularization = config['entropy_regularization']

        agent = dict(
            policy=policy, memory='recent', update=update, optimizer=optimizer, objective=objective,
            reward_estimation=reward_estimation, baseline=baseline,
            baseline_optimizer=baseline_optimizer, baseline_objective=baseline_objective,
            entropy_regularization=entropy_regularization
        )

        average_reward = list()
        final_reward = list()
        rewards = list()

        for n in range(num_runs):
            if self.num_parallel is None:
                runner = Runner(
                    agent=agent, environment=self.environment,
                    max_episode_timesteps=self.max_episode_timesteps
                )
                runner.run(num_episodes=self.num_episodes, use_tqdm=False)
            else:
                runner = Runner(
                    agent=agent, environment=self.environment,
                    max_episode_timesteps=self.max_episode_timesteps,
                    num_parallel=min(self.num_parallel, config['batch_size']),
                    remote='multiprocessing'
                )
                runner.run(
                    num_episodes=self.num_episodes, batch_agent_calls=True, sync_episodes=True,
                    use_tqdm=False
                )
            runner.close()

            average_reward.append(float(np.mean(runner.episode_returns, axis=0)))
            final_reward.append(float(np.mean(runner.episode_returns[-20:], axis=0)))
            rewards.append(list(runner.episode_returns))

        mean_average_reward = float(np.mean(average_reward, axis=0))
        mean_final_reward = float(np.mean(final_reward, axis=0))
        loss = -(mean_average_reward + mean_final_reward)

        return dict(loss=loss, info=dict(rewards=rewards))

    @staticmethod
    def get_configspace():
        configspace = cs.ConfigurationSpace()

        batch_size = cs.hyperparameters.UniformIntegerHyperparameter(
            name='batch_size', lower=1, upper=20, log=True
        )
        configspace.add_hyperparameter(hyperparameter=batch_size)

        learning_rate = cs.hyperparameters.UniformFloatHyperparameter(
            name='learning_rate', lower=1e-5, upper=1e-1, log=True
        )
        configspace.add_hyperparameter(hyperparameter=learning_rate)

        multi_step = cs.hyperparameters.UniformIntegerHyperparameter(
            name='multi_step', lower=1, upper=20, log=True
        )
        configspace.add_hyperparameter(hyperparameter=multi_step)

        horizon = cs.hyperparameters.UniformIntegerHyperparameter(
            name='horizon', lower=1, upper=100, log=True
        )
        configspace.add_hyperparameter(hyperparameter=horizon)

        discount = cs.hyperparameters.UniformFloatHyperparameter(
            name='discount', lower=0.8, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=discount)

        importance_sampling = cs.hyperparameters.CategoricalHyperparameter(
            name='importance_sampling', choices=('no', 'yes')
        )
        configspace.add_hyperparameter(hyperparameter=importance_sampling)

        # > 1.0: off (ln(1.3) roughly 1/10 of ln(5e-2))
        clipping_value = cs.hyperparameters.UniformFloatHyperparameter(
            name='clipping_value', lower=5e-2, upper=1.3, log=True
        )
        configspace.add_hyperparameter(hyperparameter=clipping_value)

        baseline = cs.hyperparameters.CategoricalHyperparameter(
            name='baseline', choices=('no', 'same', 'yes')
        )
        configspace.add_hyperparameter(hyperparameter=baseline)

        baseline_weight = cs.hyperparameters.UniformFloatHyperparameter(
            name='baseline_weight', lower=1e-2, upper=1e2
        )
        configspace.add_hyperparameter(hyperparameter=baseline_weight)

        estimate_advantage = cs.hyperparameters.CategoricalHyperparameter(
            name='estimate_advantage', choices=('no', 'yes')
        )
        configspace.add_hyperparameter(hyperparameter=estimate_advantage)

        # < 1e-5: off (ln(3e-6) roughly 1/10 of ln(1e-5))
        entropy_regularization = cs.hyperparameters.UniformFloatHyperparameter(
            name='entropy_regularization', lower=3e-6, upper=1.0, log=True
        )
        configspace.add_hyperparameter(hyperparameter=entropy_regularization)

        # configspace.add_condition(condition=cs.EqualsCondition(
        #     child=clipping_value, parent=importance_sampling, value='yes'
        # ))
        configspace.add_condition(condition=cs.NotEqualsCondition(
            child=estimate_advantage, parent=baseline, value='no'
        ))
        configspace.add_condition(condition=cs.NotEqualsCondition(
            child=baseline_weight, parent=baseline, value='no'
        ))

        return configspace



def get_kniffel_environment(env_config, config_path, env_observation_space, env_action_space, logging=False) -> Environment:
    """ Get the environment for the agent

    :return: OpenAI Gym environment
    """
    env = KniffelEnvTF(
        env_config,
        logging=logging,
        config_file_path=config_path,
        env_observation_space=env_observation_space,
        env_action_space=env_action_space,
    )

    environment = Environment.create(
        environment=env,
    )

    return environment

def main():

    # CONFIG PARAM
    runs_per_round = "1,2,5,10"
    num_iterations = 1
    selection_factor = 3
    run_id = 0
    host = nic_name_to_host(nic_name=None)
    port = 123
    max_episode_timesteps = 1000
    episodes = 1000
    restore = None
    directory = "tune_dir"

    num_parallel = 4

    # CONFIG PARAM
    config_path = "src/config/config.csv"
    base_path = str(Path(__file__).parents[2]) + "/"
    
    env_observation_space=47
    env_action_space=57

    env_config = {
        "reward_roll_dice": 0,
        "reward_game_over": -40,
        "reward_finish": 15,
        "reward_bonus": 5,
        "reward_simple": False,
    }

    environment = get_kniffel_environment(env_config, config_path, env_observation_space, env_action_space, logging=False)


    runs_per_round = tuple(int(x) for x in runs_per_round.split(','))
    print('Bayesian Optimization and Hyperband optimization')
    print(f'{num_iterations} iterations of each {len(runs_per_round)} rounds:')
    for n, num_runs in enumerate(runs_per_round, start=1):
        num_candidates = round(math.pow(selection_factor, len(runs_per_round) - n))
        print(f'round {n}: {num_candidates} candidates, each {num_runs} runs')
    print()

    server = NameServer(run_id=run_id, working_directory=directory, host=host, port=port)
    nameserver, nameserver_port = server.start()

    worker = TensorforceWorker(
        environment=environment, max_episode_timesteps=max_episode_timesteps,
        num_episodes=episodes, base=selection_factor, runs_per_round=runs_per_round,
        num_parallel=num_parallel, run_id=run_id, nameserver=nameserver,
        nameserver_port=nameserver_port, host=host
    )
    worker.run(background=True)

    if restore is None:
        previous_result = None
    else:
        previous_result = logged_results_to_HBS_result(directory=restore)

    result_logger = json_result_logger(directory=directory, overwrite=True)

    optimizer = BOHB(
        configspace=worker.get_configspace(), eta=selection_factor, min_budget=0.9,
        max_budget=math.pow(selection_factor, len(runs_per_round) - 1), run_id=run_id,
        working_directory=directory, nameserver=nameserver, nameserver_port=nameserver_port,
        host=host, result_logger=result_logger, previous_result=previous_result
    )
    # BOHB(configspace=None, eta=3, min_budget=0.01, max_budget=1, min_points_in_model=None,
    # top_n_percent=15, num_samples=64, random_fraction=1 / 3, bandwidth_factor=3,
    # min_bandwidth=1e-3, **kwargs)
    # Master(run_id, config_generator, working_directory='.', ping_interval=60,
    # nameserver='127.0.0.1', nameserver_port=None, host=None, shutdown_workers=True,
    # job_queue_sizes=(-1,0), dynamic_queue_size=True, logger=None, result_logger=None,
    # previous_result = None)
    # logger: logging.logger like object, the logger to output some (more or less meaningful)
    # information

    results = optimizer.run(n_iterations=num_iterations)
    # optimizer.run(n_iterations=1, min_n_workers=1, iteration_kwargs={})
    # min_n_workers: int, minimum number of workers before starting the run

    optimizer.shutdown(shutdown_workers=True)
    server.shutdown()

    with open(os.path.join(directory, 'results.pkl'), 'wb') as filehandle:
        pickle.dump(results, filehandle)

    print('Best found configuration: {}'.format(
        results.get_id2config_mapping()[results.get_incumbent_id()]['config']
    ))
    print('Runs:', results.get_runs_by_id(config_id=results.get_incumbent_id()))
    print('A total of {} unique configurations where sampled.'.format(
        len(results.get_id2config_mapping())
    ))
    print('A total of {} runs where executed.'.format(len(results.get_all_runs())))


if __name__ == "__main__":
    main()

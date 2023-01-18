import os
import sys
import inspect

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
    ),
)

from tensorforce import Environment

from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.dice_set import DiceSet
from src.env.env_helper import KniffelEnvHelper


class KniffelEnvTF(Environment):

    kniffel_helper: KniffelEnvHelper = None  # type: ignore
    logging = False

    state_shape = None
    action_shape = None

    def __init__(
        self,
        env_config,
        logging=False,
        env_action_space=57,
        env_observation_space=20,
        reward_mode="kniffel",  # kniffel, custom
        state_mode="binary",  # binary, continuous
    ):
        """Initialize Kniffel Envioronment"""
        self.kniffel_helper: KniffelEnvHelper = KniffelEnvHelper(
            env_config,
            logging=False,
            reward_mode=reward_mode,
            state_mode=state_mode,
        )

        self.action_shape = dict(type="int", num_values=env_action_space)
        self.observation_shape = dict(
            type="float",
            shape=(1, env_observation_space),
            min_value=-1.0,
            max_value=1.0,
        )

        self.logging = logging

        # Set start
        self.state = self.kniffel_helper.get_state()

    def states(self):
        """Return the state space of the environment

        :return: shape of the state space
        """
        return self.observation_shape

    def actions(self):
        """Return the action space of the environment

        :return: shape of the action space
        """
        return self.action_shape

    def max_episode_timesteps(self):
        """Return the maximum number of timesteps per episode

        :return: maximum number of timesteps per episode
        """
        return 1000

    def close(self):
        """Close environment"""
        super().close()

    def mock(self, dices: list):
        # print(f"Mock dice: {dices}")
        self.kniffel_helper.kniffel.mock(DiceSet(dices))

        self.state = self.kniffel_helper.get_state()

    def execute(self, actions):
        """
        Execute action

        :param actions: action to execute
        :return: state, reward, done, info
        """

        reward = 0.0

        terminal = False

        reward, terminal, _ = self.kniffel_helper.predict_and_apply(actions)

        if self.logging:
            print()
            print(f"NEW:")
            print(f"    State OLD: {self.state}")
            print(f"    Action: {actions}")

        self.state = self.kniffel_helper.get_state()

        # Return step information
        return self.state, terminal, reward

    def reset(self):
        """
        Reset state
        """
        self.kniffel_helper.reset_kniffel()
        self.state = self.kniffel_helper.get_state()

        if self.logging:
            print("RESET KNIFFEL")

        return self.state
from gym import Env
import gym.spaces as spaces
import numpy as np
from enum import Enum
import os
import sys
import inspect

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
    ),
)


from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.dice_set import DiceSet
from src.env.env_helper import KniffelEnvHelper


class KniffelEnvSB3(Env):

    kniffel_helper: KniffelEnvHelper = None  # type: ignore
    logging = False

    def __init__(
        self,
        env_config,
        logging=False,
        env_action_space=57,
        env_observation_space=47,
        reward_mode="kniffel",  # kniffel, custom
        state_mode="binary",  # binary, continuous
    ):
        """
        Kniffel environment

        Args:
            env_config (dict): Environment config
            logging (bool, optional): Enable logging. Defaults to False.
            env_action_space (int, optional): Action space. Defaults to 57.
            env_observation_space (int, optional): Observation space. Defaults to 47.
            reward_mode (str, optional): Reward mode: "kniffel" or "custom". Defaults to "kniffel".
            state_mode (str, optional): State mode: "binary" or "continuous". Defaults to "binary".
        """

        self.kniffel_helper: KniffelEnvHelper = KniffelEnvHelper(
            env_config,
            logging=False,
            reward_mode=reward_mode,
            state_mode=state_mode,
        )

        # Actions we can take
        self.action_space = spaces.Discrete(env_action_space)

        """ Example observation state
            [[ 0.66666667  0.5         0.16666667  0.66666667  0.66666667  0.33333333
            0.69230769  1.         -1.         -1.         -1.         -1.
            -1.          1.          1.          1.          1.          1.
            1.          1.          0.          0.64      ]]
        """

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1, env_observation_space), dtype=np.float16
        )

        self.logging = logging

        # Set start
        self.state = self.kniffel_helper.get_state()

    def mock(self, dices: list):
        # print(f"Mock dice: {dices}")
        self.kniffel_helper.kniffel.mock(DiceSet(dices))

        self.state = self.kniffel_helper.get_state()

    def step(self, action):
        """Apply a step to the Kniffel environment

        Args:
            action (int): Action id

        Returns:
            _type_: state, reward, done, info
        """

        reward = 0.0

        done = False

        reward, done, info = self.kniffel_helper.predict_and_apply(action)

        if self.logging:
            print()
            print(f"NEW:")
            print(f"    State OLD: {self.state}")
            print(f"    Action: {action}")

        self.state = self.kniffel_helper.get_state()

        # Return step information
        return self.state, reward, done, info

    def render(self):
        """
        Render
        """
        pass

    def reset(self):
        """
        Reset state
        """
        self.kniffel_helper.reset_kniffel()
        self.state = self.kniffel_helper.get_state()

        if self.logging:
            print("RESET KNIFFEL")

        return self.state

from gym import Env
import gym.spaces as spaces
import numpy as np
from enum import Enum
import os
import sys
import inspect
import csv

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
)


from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.dice_set import DiceSet
from src.deep_rl.env_helper import KniffelEnvHelper


class KniffelEnv(Env):

    kniffel_helper: KniffelEnvHelper = None
    logging = False

    def __init__(
        self,
        env_config,
        config_file_path="/Kniffel.CSV",
        logging=False,
        reward_roll_dice=0,
        reward_game_over=-200,
        reward_bonus=5,
        reward_finish=10,
        env_action_space=57,
        env_observation_space=20,
    ):
        """Initialize Kniffel Envioronment"""
        self.kniffel_helper = KniffelEnvHelper(
            env_config,
            logging=False,
            config_file_path=config_file_path,
            reward_roll_dice=reward_roll_dice,
            reward_game_over=reward_game_over,
            reward_bonus=reward_bonus,
            reward_finish=reward_finish,
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
            low=-1, high=1, shape=(1, env_observation_space), dtype=np.float16
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

        reward, done, _ = self.kniffel_helper.predict_and_apply(action)

        if self.logging:
            print()
            print(f"NEW:")
            print(f"    State OLD: {self.state}")
            print(f"    Action: {action}")

        self.state = self.kniffel_helper.get_state()

        # Return step information
        return self.state, reward, done, {}  # info

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

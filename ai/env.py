from tokenize import Triple
from gym import Env
import gym.spaces as spaces
import numpy as np
from enum import Enum
import os
import sys
import inspect
from sympy import elliptic_f
import tensorflow as tf

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
)

from kniffel.classes.options import KniffelOptions
from kniffel.classes.kniffel import Kniffel


class EnumAction(Enum):
    # Finish Actions
    FINISH_ONES = 0
    FINISH_TWOS = 1
    FINISH_THREES = 2
    FINISH_FOURS = 3
    FINISH_FIVES = 4
    FINISH_SIXES = 5
    FINISH_THREE_TIMES = 6
    FINISH_FOUR_TIMES = 7
    FINISH_FULL_HOUSE = 8
    FINISH_SMALL_STREET = 9
    FINISH_LARGE_STREET = 10
    FINISH_KNIFFEL = 11
    FINISH_CHANCE = 12

    # Continue Actions
    NEXT_0 = 13  # 0	    0	0	0	0	0
    NEXT_1 = 14  # 1	    0	0	0	0	1
    NEXT_2 = 15  # 2	    0	0	0	1	0
    NEXT_3 = 16  # 3	    0	0	0	1	1
    NEXT_4 = 17  # 4	    0	0	1	0	0
    NEXT_5 = 18  # 5	    0	0	1	0	1
    NEXT_6 = 19  # 6	    0	0	1	1	0
    NEXT_7 = 20  # 7	    0	0	1	1	1
    NEXT_8 = 21  # 8	    0	1	0	0	0
    NEXT_9 = 22  # 9	    0	1	0	0	1
    NEXT_10 = 23  # 10	    0	1	0	1	0
    NEXT_11 = 24  # 11	    0	1	0	1	1
    NEXT_12 = 25  # 12	    0	1	1	0	0
    NEXT_13 = 26  # 13	    0	1	1	0	1
    NEXT_14 = 27  # 14	    0	1	1	1	0
    NEXT_15 = 28  # 15	    0	1	1	1	1
    NEXT_16 = 29  # 16	    1	0	0	0	0
    NEXT_17 = 30  # 17	    1	0	0	0	1
    NEXT_18 = 31  # 18	    1	0	0	1	0
    NEXT_19 = 32  # 19	    1	0	0	1	1
    NEXT_20 = 33  # 20	    1	0	1	0	0
    NEXT_21 = 34  # 21	    1	0	1	0	1
    NEXT_22 = 35  # 22	    1	0	1	1	0
    NEXT_23 = 36  # 23	    1	0	1	1	1
    NEXT_24 = 37  # 24  	1	1	0	0	0
    NEXT_25 = 38  # 25  	1	1	0	0	1
    NEXT_26 = 39  # 26  	1	1	0	1	0
    NEXT_27 = 40  # 27  	1	1	0	1	1
    NEXT_28 = 41  # 28  	1	1	1	0	0
    NEXT_29 = 42  # 29  	1	1	1	0	1
    NEXT_30 = 43  # 30  	1	1	1	1	0
    NEXT_31 = 44  # 31  	1	1	1	1	1


class KniffelEnv(Env):
    def __init__(
        self,
        reward_step=0,
        reward_round=0.5,
        reward_roll_dice=0.25,
        reward_game_over=-10,
        reward_bonus=2,
        reward_finish=10,
        reward_zero_dice=-0.5,
        reward_one_dice=-0.2,
        reward_twos_dice=-0.1,
        reward_three_dice=0.5,
        reward_four_dice=0.6,
        reward_five_dice=0.8,
        reward_six_dice=1,
    ):
        """Initialize Kniffel Envioronment

        :param reward_step: Reward for a normal step, defaults to -0.5
        :param reward_round: Reward for finishing a round, defaults to 5
        :param reward_roll_dice: Reward for rolling a dice, defaults to 5
        :param reward_game_over: Reward for failing a game, defaults to -10
        :param reward_bonus: Reward if bonus received, defaults to 2
        :param reward_finish: Reward if game finished, defaults to 10
        :param reward_zero_dice: Reward if zero dices used, defaults to -0.5
        :param reward_one_dice: Reward if one dices used, defaults to -0.2
        :param reward_twos_dice: Reward if two dices used, defaults to -0.1
        :param reward_three_dice: Reward if three dices used, defaults to 5
        :param reward_four_dice: Reward if four dices used, defaults to 5.33
        :param reward_five_dice: Reward if five dices used, defaults to 5.66
        :param reward_six_dice: Reward if six dices used, defaults to 6
        """
        self.kniffel = Kniffel()
        # Actions we can take
        self.action_space = spaces.Discrete(44)

        self.observation_space = spaces.Box(
            low=0, high=13, shape=(13, 16), dtype=np.int32
        )

        # Set start
        self.state = self.kniffel.get_array()

        self._reward_step = reward_step
        self._reward_round = reward_round
        self._reward_roll_dice = reward_roll_dice
        self._reward_game_over = reward_game_over
        self._reward_bonus = reward_bonus
        self._reward_finish = reward_finish

        self._reward_zero_dice = reward_zero_dice
        self._reward_one_dice = reward_one_dice
        self._reward_two_dice = reward_twos_dice
        self._reward_three_dice = reward_three_dice
        self._reward_four_dice = reward_four_dice
        self._reward_five_dice = reward_five_dice
        self._reward_six_dice = reward_six_dice

    def rewards_calculator(self, dice_count) -> float:
        """Calculate reward based on amount of dices used for finishing the round.

        :param dice_count: amount of dices
        :return: reward
        """
        if dice_count == 0:
            return self._reward_zero_dice
        elif dice_count == 1:
            return self._reward_one_dice
        elif dice_count == 2:
            return self._reward_two_dice
        elif dice_count == 3:
            return self._reward_three_dice
        elif dice_count == 4:
            return self._reward_four_dice
        elif dice_count == 5:
            return self._reward_five_dice
        elif dice_count == 6:
            return self._reward_six_dice

    def step(self, action):
        reward = 0
        has_bonus = self.kniffel.is_bonus()

        done = False
        # Apply action
        enum_action = EnumAction(action)
        try:
            # Finish Actions
            if EnumAction.FINISH_ONES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.ONES) / 1
                reward += self.rewards_calculator(points)
            if EnumAction.FINISH_TWOS is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.TWOS) / 2
                reward += self.rewards_calculator(points)
            if EnumAction.FINISH_THREES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.THREES) / 3
                reward += self.rewards_calculator(points)
            if EnumAction.FINISH_FOURS is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FOURS) / 4
                reward += self.rewards_calculator(points)
            if EnumAction.FINISH_FIVES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FIVES) / 5
                reward += self.rewards_calculator(points)
            if EnumAction.FINISH_SIXES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.SIXES) / 6
                reward += self.rewards_calculator(points)
            if EnumAction.FINISH_THREE_TIMES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.THREE_TIMES)
                reward += points / 30
            if EnumAction.FINISH_FOUR_TIMES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FOUR_TIMES)
                reward += points / 30
            if EnumAction.FINISH_FULL_HOUSE is enum_action:
                self.kniffel.finish_turn(KniffelOptions.FULL_HOUSE)
                reward += self._reward_round
            if EnumAction.FINISH_SMALL_STREET is enum_action:
                self.kniffel.finish_turn(KniffelOptions.SMALL_STREET)
                reward += self._reward_round
            if EnumAction.FINISH_LARGE_STREET is enum_action:
                self.kniffel.finish_turn(KniffelOptions.LARGE_STREET)
                reward += self._reward_round
            if EnumAction.FINISH_KNIFFEL is enum_action:
                self.kniffel.finish_turn(KniffelOptions.KNIFFEL)
                reward += self._reward_round
            if EnumAction.FINISH_CHANCE is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.CHANCE)
                reward += points / 30

            # Continue enum_actions
            if EnumAction.NEXT_0 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_1 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_2 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_3 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_4 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_5 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_6 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_7 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_8 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_9 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_10 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_11 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_12 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_13 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_14 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_15 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_16 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_17 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_18 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_19 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_20 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_21 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_22 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_23 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_24 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_25 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_26 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_27 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_28 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_29 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_30 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_31 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 1])
                reward += self._reward_roll_dice

            if (
                self.kniffel.is_bonus()
                and has_bonus is False
                and (
                    EnumAction.FINISH_ONES is enum_action
                    or EnumAction.FINISH_ONES is enum_action
                    or EnumAction.FINISH_TWOS is enum_action
                    or EnumAction.FINISH_THREES is enum_action
                    or EnumAction.FINISH_FOURS is enum_action
                    or EnumAction.FINISH_FIVES is enum_action
                    or EnumAction.FINISH_SIXES is enum_action
                )
            ):
                reward += self._reward_bonus

        except Exception as e:
            reward += self._reward_game_over
            done = True

        self.state = self.kniffel.get_array()

        # Check if shower is done
        if self.kniffel.is_finished():
            reward += self._reward_finish
            done = True
        elif done is False:
            done = False

        # Set placeholder for info
        info = {}

        reward += self._reward_step

        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset
        del self.state
        del self.kniffel

        self.kniffel = Kniffel()
        self.state = self.kniffel.get_array()

        return self.state

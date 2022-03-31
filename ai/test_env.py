from tokenize import Triple
from gym import Env
import gym.spaces as spaces
import numpy as np
from enum import Enum
import os
import sys
import inspect
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
    PENALTY = -375
    REWARD = 10

    def __init__(self):
        self.kniffel = Kniffel()
        # Actions we can take
        self.action_space = spaces.Discrete(44)

        self.observation_space = spaces.Box(
            low=0, high=13, shape=(13, 16), dtype=np.int32
        )

        # Set start
        self.state = self.kniffel.get_array()

    def step(self, action):
        done = False
        # Apply action
        enum_action = EnumAction(action)
        try:
            points = 0
            # Finish Actions
            if EnumAction.FINISH_ONES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.ONES)
            if EnumAction.FINISH_TWOS is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.TWOS)
            if EnumAction.FINISH_THREES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.THREES)
            if EnumAction.FINISH_FOURS is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FOURS)
            if EnumAction.FINISH_FIVES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FIVES)
            if EnumAction.FINISH_SIXES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.SIXES)
            if EnumAction.FINISH_THREE_TIMES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.THREE_TIMES)
            if EnumAction.FINISH_FOUR_TIMES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FOUR_TIMES)
            if EnumAction.FINISH_FULL_HOUSE is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FULL_HOUSE)
            if EnumAction.FINISH_SMALL_STREET is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.SMALL_STREET)
            if EnumAction.FINISH_LARGE_STREET is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.LARGE_STREET)
            if EnumAction.FINISH_KNIFFEL is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.KNIFFEL)
            if EnumAction.FINISH_CHANCE is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.CHANCE)

            # Continue enum_actions
            if EnumAction.NEXT_0 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 0])
            if EnumAction.NEXT_1 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 1])
            if EnumAction.NEXT_2 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 0])
            if EnumAction.NEXT_3 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 1])
            if EnumAction.NEXT_4 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 0])
            if EnumAction.NEXT_5 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 1])
            if EnumAction.NEXT_6 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 0])
            if EnumAction.NEXT_7 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 1])
            if EnumAction.NEXT_8 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 0])
            if EnumAction.NEXT_9 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 1])
            if EnumAction.NEXT_10 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 0])
            if EnumAction.NEXT_11 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 1])
            if EnumAction.NEXT_12 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 0])
            if EnumAction.NEXT_13 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 1])
            if EnumAction.NEXT_14 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 0])
            if EnumAction.NEXT_15 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 1])
            if EnumAction.NEXT_16 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 0])
            if EnumAction.NEXT_17 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 1])
            if EnumAction.NEXT_18 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 0])
            if EnumAction.NEXT_19 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 1])
            if EnumAction.NEXT_20 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 0])
            if EnumAction.NEXT_21 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 1])
            if EnumAction.NEXT_22 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 0])
            if EnumAction.NEXT_23 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 1])
            if EnumAction.NEXT_24 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 0])
            if EnumAction.NEXT_25 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 1])
            if EnumAction.NEXT_26 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 0])
            if EnumAction.NEXT_27 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 1])
            if EnumAction.NEXT_28 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 0])
            if EnumAction.NEXT_29 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 1])
            if EnumAction.NEXT_30 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 0])
            if EnumAction.NEXT_31 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 1])

            print("#Rewards: " + str(self.kniffel.get_points()))
            reward = self.REWARD + self.kniffel.get_points()
        except Exception as e:
            # print(e)
            reward = self.PENALTY  # - self.kniffel.get_points()
            done = True

        self.state = self.kniffel.get_array()

        # Check if shower is done
        if self.kniffel.is_finished():
            done = True
        elif done is False:
            done = False

        # Set placeholder for info
        info = {}

        # Return step information
        return self.state, reward, done, info


if __name__ == "__main__":
    env = KniffelEnv()

    n_state, reward, done, info = env.step(EnumAction.NEXT_27)
    print(done)
    n_state, reward, done, info = env.step(EnumAction.NEXT_30)

    print(done)
    n_state, reward, done, info = env.step(EnumAction.NEXT_19)
    print(done)
    n_state, reward, done, info = env.step(EnumAction.FINISH_CHANCE)
    print(done)
    n_state, reward, done, info = env.step(EnumAction.NEXT_27)
    print(done)
    n_state, reward, done, info = env.step(EnumAction.FINISH_FIVES)

    print(done)
    print(reward)
    print(n_state)

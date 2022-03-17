import os
import re
import sys
import inspect
from time import process_time_ns

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from enum import Enum
from kniffel.classes.options import KniffelOptions
from kniffel.classes.kniffel import Kniffel
from enum import Enum
import random


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


class AI:
    done: bool = False

    REWARD: int = 10
    PENALTY: int = -100

    def __init__(self) -> None:
        self.kniffel = Kniffel()

    def is_done(self):
        if self.done is False:
            self.done = self.kniffel.is_finished()
        return self.done

    def step(self, action: EnumAction):
        """
        Run the desired action based on the action enum

        :param EnumAction action: Action what to do
        """

        try:
            # Finish Actions
            if action.FINISH_ONES is action:
                self.kniffel.finish_turn(KniffelOptions.ONES)
            if action.FINISH_TWOS is action:
                self.kniffel.finish_turn(KniffelOptions.TWOS)
            if action.FINISH_THREES is action:
                self.kniffel.finish_turn(KniffelOptions.THREES)
            if action.FINISH_FOURS is action:
                self.kniffel.finish_turn(KniffelOptions.FOURS)
            if action.FINISH_FIVES is action:
                self.kniffel.finish_turn(KniffelOptions.FIVES)
            if action.FINISH_SIXES is action:
                self.kniffel.finish_turn(KniffelOptions.SIXES)
            if action.FINISH_THREE_TIMES is action:
                self.kniffel.finish_turn(KniffelOptions.THREE_TIMES)
            if action.FINISH_FOUR_TIMES is action:
                self.kniffel.finish_turn(KniffelOptions.FOUR_TIMES)
            if action.FINISH_FULL_HOUSE is action:
                self.kniffel.finish_turn(KniffelOptions.FULL_HOUSE)
            if action.FINISH_SMALL_STREET is action:
                self.kniffel.finish_turn(KniffelOptions.SMALL_STREET)
            if action.FINISH_LARGE_STREET is action:
                self.kniffel.finish_turn(KniffelOptions.LARGE_STREET)
            if action.FINISH_KNIFFEL is action:
                self.kniffel.finish_turn(KniffelOptions.KNIFFEL)
            if action.FINISH_CHANCE is action:
                self.kniffel.finish_turn(KniffelOptions.CHANCE)

            # Continue Actions
            if action.NEXT_0 is action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 0])
            if action.NEXT_1 is action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 1])
            if action.NEXT_2 is action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 0])
            if action.NEXT_3 is action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 1])
            if action.NEXT_4 is action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 0])
            if action.NEXT_5 is action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 1])
            if action.NEXT_6 is action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 0])
            if action.NEXT_7 is action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 1])
            if action.NEXT_8 is action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 0])
            if action.NEXT_9 is action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 1])
            if action.NEXT_10 is action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 0])
            if action.NEXT_11 is action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 1])
            if action.NEXT_12 is action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 0])
            if action.NEXT_13 is action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 1])
            if action.NEXT_14 is action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 0])
            if action.NEXT_15 is action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 1])
            if action.NEXT_16 is action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 0])
            if action.NEXT_17 is action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 1])
            if action.NEXT_18 is action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 0])
            if action.NEXT_19 is action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 1])
            if action.NEXT_20 is action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 0])
            if action.NEXT_21 is action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 1])
            if action.NEXT_22 is action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 0])
            if action.NEXT_23 is action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 1])
            if action.NEXT_24 is action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 0])
            if action.NEXT_25 is action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 1])
            if action.NEXT_26 is action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 0])
            if action.NEXT_27 is action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 1])
            if action.NEXT_28 is action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 0])
            if action.NEXT_29 is action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 1])
            if action.NEXT_30 is action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 0])
            if action.NEXT_31 is action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 1])
        except:
            print("Error")
            self.done = True
            return True, self.PENALTY

        return self.done, self.REWARD


class State:
    points = 0
    turn = 0  # amount of turns done already
    attempt = 0  # amount of attempts tried in the current turn

    selected_options: KniffelOptions = None
    possible_options: KniffelOptions = None


def play(n: int):
    for i in range(n):
        print(f"Game {i+1}/{n}")
        ai = AI()
        done = False

        tot_rewards = 0

        while done is False:
            # do random action
            done, reward = ai.step(EnumAction(random.randint(0, 44)))
            tot_rewards += reward

        print(tot_rewards)
        ai.kniffel.print()


if __name__ == "__main__":
    play(5)

# Goal:
#   So nah wie möglich an die maximal Punkzahl 375 zu kommen

# Penalty:
#   Wenn nicht möglicher move erfolgt

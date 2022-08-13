from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.options import KniffelOptions

import numpy as np


def play():

    try:
        kniffel = Kniffel(True)

        kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
        kniffel.finish_turn(KniffelOptions.THREE_TIMES)

        kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
        kniffel.finish_turn(KniffelOptions.FOUR_TIMES)

        kniffel.mock(DiceSet([6, 6, 6, 5, 5]))
        kniffel.finish_turn(KniffelOptions.FULL_HOUSE)

        kniffel.mock(DiceSet([1, 2, 3, 4, 5]))
        kniffel.finish_turn(KniffelOptions.SMALL_STREET)

        kniffel.mock(DiceSet([1, 2, 3, 4, 5]))
        kniffel.finish_turn(KniffelOptions.LARGE_STREET)

        print(kniffel.get_state())
        print()
        print(np.shape(kniffel.get_state()))

        kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
        kniffel.finish_turn(KniffelOptions.KNIFFEL)

        kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
        kniffel.finish_turn(KniffelOptions.CHANCE)

        kniffel.mock(DiceSet([1, 1, 1, 1, 1]))
        kniffel.mock(DiceSet([1, 1, 1, 1, 1]))
        kniffel.finish_turn(KniffelOptions.ONES)

        kniffel.mock(DiceSet([2, 2, 2, 2, 2]))
        kniffel.finish_turn(KniffelOptions.TWOS)

        kniffel.mock(DiceSet([3, 3, 3, 3, 3]))
        kniffel.finish_turn(KniffelOptions.THREES)

        kniffel.mock(DiceSet([4, 4, 4, 4, 4]))
        kniffel.finish_turn(KniffelOptions.FOURS)

        kniffel.mock(DiceSet([5, 5, 5, 5, 5]))
        kniffel.finish_turn(KniffelOptions.FIVES)

        kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
        kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
        kniffel.finish_turn(KniffelOptions.SIXES)

    except Exception as e:
        print(e)
        print(kniffel.get_state())
        print()
        print(np.shape(kniffel.get_state()))


if __name__ == "__main__":
    play()

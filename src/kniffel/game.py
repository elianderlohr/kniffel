# from click import option
from classes.kniffel import Kniffel
from classes.dice_set import DiceSet
from classes.options import KniffelOptions

import numpy as np


def play():
    kniffel = Kniffel(True)

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
    kniffel.finish_turn(KniffelOptions.THREE_TIMES)

    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.SIXES)

    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.FOUR_TIMES)

    print(kniffel.is_finished())

    kniffel.mock(DiceSet([6, 6, 6, 5, 5]))
    kniffel.finish_turn(KniffelOptions.FULL_HOUSE)

    kniffel.mock(DiceSet([1, 2, 3, 4, 5]))
    kniffel.finish_turn(KniffelOptions.SMALL_STREET)

    kniffel.mock(DiceSet([1, 2, 3, 4, 5]))
    kniffel.finish_turn(KniffelOptions.LARGE_STREET)

    kniffel.mock(DiceSet([6, 6, 6, 6, 5]))
    kniffel.finish_turn(KniffelOptions.KNIFFEL_SLASH)

    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    # kniffel.finish_turn(KniffelOptions.CHANCE)

    print()
    kniffel.print()

    print(kniffel.is_finished())
    print(kniffel.get_state())
    print()
    print(np.shape(kniffel.get_state()))


if __name__ == "__main__":
    play()
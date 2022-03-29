from click import option
from classes.kniffel import Kniffel
from classes.dice_set import DiceSet
from classes.options import KniffelOptions
from classes.kniffel_check import KniffelCheck
import numpy as np


def main():
    kniffel = Kniffel(False)

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
    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.SIXES)

    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.FOUR_TIMES)

    kniffel.mock(DiceSet([6, 6, 6, 5, 5]))
    kniffel.finish_turn(KniffelOptions.FULL_HOUSE)

    kniffel.mock(DiceSet([1, 2, 3, 4, 5]))
    kniffel.finish_turn(KniffelOptions.SMALL_STREET)

    kniffel.mock(DiceSet([1, 2, 3, 4, 5]))
    kniffel.finish_turn(KniffelOptions.LARGE_STREET)

    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.KNIFFEL)

    kniffel.mock(DiceSet([6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.CHANCE)

    # kniffel.add_turn([1, 1, 1, 1, 1])

    kniffel.print()
    print(kniffel.get_points())
    print(kniffel.is_bonus())
    d = kniffel.get_array()

    print(d)


def test2():
    kniffel = Kniffel(False)
    d = kniffel.get_array()
    print(d.shape)
    print(d)


def test3():

    test = KniffelCheck().check_chance(DiceSet([1, 1, 1, 1, 1]))
    print(test)


def test4():
    option = KniffelOptions.ONES

    selected = None

    if option is KniffelOptions.ONES:
        print("##ones")
        selected = KniffelCheck().check_1(DiceSet([1, 1, 1, 1, 1]))

    if option is KniffelOptions.TWOS:
        print("##twos")
        selected = KniffelCheck().check_2(DiceSet([1, 1, 1, 1, 1]))

    if option is KniffelOptions.THREES:
        selected = KniffelCheck().check_3(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.FOURS:
        selected = KniffelCheck().check_4(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.FIVES:
        selected = KniffelCheck().check_5(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.SIXES:
        selected = KniffelCheck().check_6(DiceSet([1, 1, 1, 1, 1]))

    if option is KniffelOptions.THREE_TIMES:
        selected = KniffelCheck().check_three_times(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.FOUR_TIMES:
        selected = KniffelCheck().check_four_times(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.FULL_HOUSE:
        selected = KniffelCheck().check_full_house(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.SMALL_STREET:
        selected = KniffelCheck().check_small_street(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.LARGE_STREET:
        selected = KniffelCheck().check_large_street(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.KNIFFEL:
        selected = KniffelCheck().check_kniffel(DiceSet([1, 1, 1, 1, 1]))
    if option is KniffelOptions.CHANCE:
        selected = KniffelCheck().check_chance(DiceSet([1, 1, 1, 1, 1]))

    print("## " + str(selected))


if __name__ == "__main__":
    test4()

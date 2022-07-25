import pytest

from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.kniffel_check import KniffelCheck
from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.kniffel_option import KniffelOptionClass
from src.kniffel.classes.options import KniffelOptions


def test_what_occures_n_times():
    """Check that 1 occure 3, 2 or 1 times"""
    dice_set = DiceSet(mock=[1, 1, 1, 2, 2])

    assert KniffelCheck().what_occures_n_times(dice_set, 3) == 1
    assert KniffelCheck().what_occures_n_times(dice_set, 2) == 1
    assert KniffelCheck().what_occures_n_times(dice_set, 1) == 1


def test_occures_n_times():
    """Do some dices occure 3 time and some twice (not ones)"""
    dice_set = DiceSet(mock=[1, 1, 1, 2, 2])

    assert KniffelCheck().occures_n_times(dice_set, 3, []) == True
    assert KniffelCheck().occures_n_times(dice_set, 2, [1]) == True


# NORMAL
def test_check_1():
    dice_set = DiceSet(mock=[1, 1, 1, 1, 6])

    kniffel_options = KniffelCheck().check_1(ds=dice_set)

    assert kniffel_options.name == "ones"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 4
    assert kniffel_options.id == KniffelOptions.ONES.value


def test_check_2():
    dice_set = DiceSet(mock=[2, 2, 2, 2, 6])

    kniffel_options = KniffelCheck().check_2(ds=dice_set)

    assert kniffel_options.name == "twos"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 8
    assert kniffel_options.id == KniffelOptions.TWOS.value


def test_check_3():
    dice_set = DiceSet(mock=[3, 3, 3, 3, 6])

    kniffel_options = KniffelCheck().check_3(ds=dice_set)

    assert kniffel_options.name == "threes"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 12
    assert kniffel_options.id == KniffelOptions.THREES.value


def test_check_4():
    dice_set = DiceSet(mock=[4, 4, 4, 4, 6])

    kniffel_options = KniffelCheck().check_4(ds=dice_set)

    assert kniffel_options.name == "fours"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 16
    assert kniffel_options.id == KniffelOptions.FOURS.value


def test_check_5():
    dice_set = DiceSet(mock=[5, 5, 5, 5, 6])

    kniffel_options = KniffelCheck().check_5(ds=dice_set)

    assert kniffel_options.name == "fives"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 20
    assert kniffel_options.id == KniffelOptions.FIVES.value


def test_check_6():
    dice_set = DiceSet(mock=[6, 6, 6, 6, 5])

    kniffel_options = KniffelCheck().check_6(ds=dice_set)

    assert kniffel_options.name == "sixes"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 24
    assert kniffel_options.id == KniffelOptions.SIXES.value


# THREE TIMES
def test_check_three_times():
    dice_set = DiceSet(mock=[6, 6, 6, 4, 5])

    kniffel_options = KniffelCheck().check_three_times(ds=dice_set)

    assert kniffel_options.name == "three-times"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 27
    assert kniffel_options.id == KniffelOptions.THREE_TIMES.value


def test_check_three_times_negative():
    dice_set = DiceSet(mock=[6, 6, 4, 4, 5])

    kniffel_options = KniffelCheck().check_three_times(ds=dice_set)

    assert kniffel_options.name == "three-times"
    assert kniffel_options.is_possible == False
    assert kniffel_options.points == 0
    assert kniffel_options.id == KniffelOptions.THREE_TIMES.value


# FOUR TIMES
def test_check_four_times():
    dice_set = DiceSet(mock=[6, 6, 6, 6, 5])

    kniffel_options = KniffelCheck().check_four_times(ds=dice_set)

    assert kniffel_options.name == "four-times"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 29
    assert kniffel_options.id == KniffelOptions.FOUR_TIMES.value


def test_check_four_times_negative():
    dice_set = DiceSet(mock=[6, 6, 6, 5, 5])

    kniffel_options = KniffelCheck().check_four_times(ds=dice_set)

    assert kniffel_options.name == "four-times"
    assert kniffel_options.is_possible == False
    assert kniffel_options.points == 0
    assert kniffel_options.id == KniffelOptions.FOUR_TIMES.value


# FULL HOUSE
def test_check_full_house_times():
    dice_set = DiceSet(mock=[6, 6, 6, 5, 5])

    kniffel_options = KniffelCheck().check_full_house(ds=dice_set)

    assert kniffel_options.name == "full-house"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 25
    assert kniffel_options.id == KniffelOptions.FULL_HOUSE.value


def test_check_full_house_negative():
    dice_set = DiceSet(mock=[6, 6, 6, 6, 5])

    kniffel_options = KniffelCheck().check_full_house(ds=dice_set)

    assert kniffel_options.name == "full-house"
    assert kniffel_options.is_possible == False
    assert kniffel_options.points == 0
    assert kniffel_options.id == KniffelOptions.FULL_HOUSE.value


# SMALL STREET
def test_check_small_street_times_option_1():
    dice_set = DiceSet(mock=[1, 2, 3, 4, 6])

    kniffel_options = KniffelCheck().check_small_street(ds=dice_set)

    assert kniffel_options.name == "small-street"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 30
    assert kniffel_options.id == KniffelOptions.SMALL_STREET.value


def test_check_small_street_times_option_2():
    dice_set = DiceSet(mock=[6, 2, 3, 4, 5])

    kniffel_options = KniffelCheck().check_small_street(ds=dice_set)

    assert kniffel_options.name == "small-street"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 30
    assert kniffel_options.id == KniffelOptions.SMALL_STREET.value


def test_check_small_street_times_option_3():
    dice_set = DiceSet(mock=[1, 2, 3, 4, 5])

    kniffel_options = KniffelCheck().check_small_street(ds=dice_set)

    assert kniffel_options.name == "small-street"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 30
    assert kniffel_options.id == KniffelOptions.SMALL_STREET.value


def test_check_small_street_times_negative():
    dice_set = DiceSet(mock=[1, 2, 3, 3, 3])

    kniffel_options = KniffelCheck().check_small_street(ds=dice_set)

    assert kniffel_options.name == "small-street"
    assert kniffel_options.is_possible == False
    assert kniffel_options.points == 0
    assert kniffel_options.id == KniffelOptions.SMALL_STREET.value


# LARGE STREET
def test_check_large_street_times_option_1():
    dice_set = DiceSet(mock=[1, 2, 3, 4, 5])

    kniffel_options = KniffelCheck().check_large_street(ds=dice_set)

    assert kniffel_options.name == "large-street"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 40
    assert kniffel_options.id == KniffelOptions.LARGE_STREET.value


def test_check_large_street_times_negative():
    dice_set = DiceSet(mock=[1, 2, 3, 3, 3])

    kniffel_options = KniffelCheck().check_large_street(ds=dice_set)

    assert kniffel_options.name == "large-street"
    assert kniffel_options.is_possible == False
    assert kniffel_options.points == 0
    assert kniffel_options.id == KniffelOptions.LARGE_STREET.value


# KNIFFEL
def test_check_kniffel():
    dice_set = DiceSet(mock=[1, 1, 1, 1, 1])

    kniffel_options = KniffelCheck().check_kniffel(ds=dice_set)

    assert kniffel_options.name == "kniffel"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 50
    assert kniffel_options.id == KniffelOptions.KNIFFEL.value


def test_check_kniffel_negative():
    dice_set = DiceSet(mock=[1, 2, 3, 3, 3])

    kniffel_options = KniffelCheck().check_kniffel(ds=dice_set)

    assert kniffel_options.name == "kniffel"
    assert kniffel_options.is_possible == False
    assert kniffel_options.points == 0
    assert kniffel_options.id == KniffelOptions.KNIFFEL.value


# CHANCE
def test_check_chance():
    dice_set = DiceSet(mock=[1, 1, 1, 1, 1])

    kniffel_options = KniffelCheck().check_chance(ds=dice_set)

    assert kniffel_options.name == "chance"
    assert kniffel_options.is_possible == True
    assert kniffel_options.points == 5
    assert kniffel_options.id == KniffelOptions.CHANCE.value

import pytest

from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.options import KniffelOptions
from src.kniffel.classes.kniffel import Kniffel


def test_wrong_option():
    kniffel = Kniffel()

    kniffel.mock(DiceSet(mock=[1, 1, 1, 1, 1]))
    kniffel.finish_turn(KniffelOptions.ONES)

    kniffel.mock(
        DiceSet(
            mock=[
                2,
                2,
                2,
                2,
                2,
            ]
        )
    )

    try:
        kniffel.finish_turn(KniffelOptions.THREES)
    except Exception as e:
        assert (
            e.args[0]
            == "Cannot select the same Option again or not possible for this. Select another Option!"
        )


def test_option_selected_twice():
    kniffel = Kniffel()

    kniffel.mock(DiceSet(mock=[1, 1, 1, 1, 1]))
    kniffel.finish_turn(KniffelOptions.ONES)

    kniffel.mock(DiceSet(mock=[1, 1, 1, 1, 1]))

    try:
        kniffel.finish_turn(KniffelOptions.ONES)
    except Exception as e:
        assert (
            e.args[0]
            == "Cannot select the same Option again or not possible for this. Select another Option!"
        )


def test_two_many_attempts():
    kniffel = Kniffel()

    kniffel.mock(DiceSet(mock=[1, 1, 1, 1, 1]))
    kniffel.finish_turn(KniffelOptions.ONES)

    try:
        kniffel.mock(DiceSet(mock=[1, 1, 1, 1, 1]))
        kniffel.add_turn(keep=[1, 1, 1, 1, 1])
        kniffel.add_turn(keep=[1, 1, 1, 1, 1])
        kniffel.add_turn(keep=[1, 1, 1, 1, 1])

    except Exception as e:
        assert e.args[0] == "Cannot do more then 3 attempts per round."


def test_finish_game():
    kniffel = Kniffel()

    kniffel.mock(DiceSet(mock=[1, 1, 1, 1, 1]))
    kniffel.finish_turn(KniffelOptions.ONES)

    kniffel.mock(
        DiceSet(
            mock=[
                2,
                2,
                2,
                2,
                2,
            ]
        )
    )
    kniffel.finish_turn(KniffelOptions.TWOS)

    kniffel.mock(DiceSet(mock=[3, 3, 3, 3, 3]))
    kniffel.finish_turn(KniffelOptions.THREES)

    kniffel.mock(DiceSet(mock=[4, 4, 4, 4, 4]))
    kniffel.finish_turn(KniffelOptions.FOURS)

    kniffel.mock(
        DiceSet(
            mock=[
                5,
                5,
                5,
                5,
                5,
            ]
        )
    )
    kniffel.finish_turn(KniffelOptions.FIVES)

    kniffel.mock(DiceSet(mock=[6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.SIXES)

    kniffel.mock(DiceSet(mock=[6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.THREE_TIMES)

    kniffel.mock(DiceSet(mock=[6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.FOUR_TIMES)

    kniffel.mock(DiceSet(mock=[6, 6, 6, 5, 5]))
    kniffel.finish_turn(KniffelOptions.FULL_HOUSE)

    kniffel.mock(DiceSet(mock=[1, 2, 3, 4, 5]))
    kniffel.finish_turn(KniffelOptions.SMALL_STREET)

    kniffel.mock(DiceSet(mock=[1, 2, 3, 4, 5]))
    kniffel.finish_turn(KniffelOptions.LARGE_STREET)

    kniffel.mock(DiceSet(mock=[6, 6, 6, 6, 6]))
    kniffel.finish_turn(KniffelOptions.KNIFFEL)

    kniffel.mock(DiceSet(mock=[6, 6, 6, 6, 6]))

    try:
        kniffel.finish_turn(KniffelOptions.CHANCE)
    except Exception as e:
        assert e.args[0] == "Game finished!"

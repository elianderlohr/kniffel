import pytest

from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.attempt import Attempt
from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.options import KniffelOptions
from src.kniffel.classes.status import KniffelStatus


def test_keep():
    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    attempt.add_attempt(keep=[0, 0, 1, 1, 1])

    assert attempt.get_latest().get_dice(1).value in [1, 2, 3, 4, 5, 6]
    assert attempt.get_latest().get_dice(2).value in [1, 2, 3, 4, 5, 6]
    assert attempt.get_latest().get_dice(3).value == 3
    assert attempt.get_latest().get_dice(4).value == 3
    assert attempt.get_latest().get_dice(5).value == 3


def test_status():
    attempt = Attempt()
    assert attempt.status.INIT == KniffelStatus.INIT

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.status.ATTEMPTING == KniffelStatus.ATTEMPTING
    assert attempt.count() == 1

    attempt.add_attempt(keep=[0, 0, 1, 1, 1])

    assert attempt.count() == 2

    attempt.mock(mock=DiceSet(mock=[1, 1, 1, 1, 1]))

    assert attempt.count() == 3

    option = attempt.finish_attempt(KniffelOptions.KNIFFEL)

    assert attempt.status.FINISHED == KniffelStatus.FINISHED
    assert option.points == 50


def test_one_finish():
    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.count() == 1

    attempt.finish_attempt(KniffelOptions.THREE_TIMES)

    assert attempt.status == KniffelStatus.FINISHED


def test_two_finish():
    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.count() == 1

    attempt.add_attempt(keep=[1, 1, 1, 1, 1])

    assert attempt.count() == 2

    attempt.finish_attempt(KniffelOptions.THREE_TIMES)

    assert attempt.status == KniffelStatus.FINISHED


def test_three_finish():
    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.count() == 1

    attempt.add_attempt(keep=[1, 1, 1, 1, 1])

    assert attempt.count() == 2

    attempt.add_attempt(keep=[1, 1, 1, 1, 1])

    assert attempt.count() == 3

    attempt.finish_attempt(KniffelOptions.THREE_TIMES)

    assert attempt.status == KniffelStatus.FINISHED


def test_four_attempts():
    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.count() == 1

    attempt.add_attempt(keep=[0, 0, 1, 1, 1])

    assert attempt.count() == 2

    attempt.mock(mock=DiceSet(mock=[1, 1, 1, 1, 1]))

    assert attempt.count() == 3
    assert attempt.status.ATTEMPTING == KniffelStatus.ATTEMPTING

    try:
        attempt.finish_attempt(KniffelOptions.KNIFFEL)

        attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))
    except Exception as e:
        assert e.args[0] == "Cannot do more then 3 attempts per round."


def test_wrong_finish():
    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.count() == 1

    attempt.add_attempt(keep=[0, 0, 1, 1, 1])

    assert attempt.count() == 2

    attempt.mock(mock=DiceSet(mock=[1, 1, 1, 1, 1]))

    assert attempt.count() == 3

    option = attempt.finish_attempt(KniffelOptions.FULL_HOUSE)

    assert option.is_possible == False
    assert option.points == 0


def test_slash():
    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.count() == 1

    attempt.add_attempt(keep=[0, 0, 1, 1, 1])

    assert attempt.count() == 2

    attempt.mock(mock=DiceSet(mock=[1, 1, 1, 1, 1]))

    assert attempt.count() == 3

    option = attempt.finish_attempt(KniffelOptions.SMALL_STREET_SLASH)

    assert option.is_possible == True
    assert option.points == 0
    assert option.id == KniffelOptions.SMALL_STREET_SLASH.value


def test_slashing_possible():
    """Test to slash a possible combination"""

    attempt = Attempt()

    attempt.add_attempt(dice_set=DiceSet(mock=[1, 2, 3, 3, 3]))

    assert attempt.count() == 1

    attempt.add_attempt(keep=[0, 0, 1, 1, 1])

    assert attempt.count() == 2

    attempt.mock(mock=DiceSet(mock=[1, 1, 1, 1, 1]))

    assert attempt.count() == 3

    option = attempt.finish_attempt(KniffelOptions.KNIFFEL_SLASH)

    assert option.is_possible == True
    assert option.points == 50
    assert option.id == KniffelOptions.KNIFFEL.value

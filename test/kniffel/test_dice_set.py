import pytest

from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.dice_set import DiceSet


def test_random_dice_set():
    """Test random dice set throws and check if 5 dices and each dice is between the value of 1 and 6"""

    dice_set = DiceSet()

    assert len(dice_set.get().items()) == 5

    assert dice_set.get()[1].get() in [1, 2, 3, 4, 5, 6]
    assert dice_set.get()[2].get() in [1, 2, 3, 4, 5, 6]
    assert dice_set.get()[3].get() in [1, 2, 3, 4, 5, 6]
    assert dice_set.get()[4].get() in [1, 2, 3, 4, 5, 6]
    assert dice_set.get()[5].get() in [1, 2, 3, 4, 5, 6]


def test_get_dice():
    dice_set = DiceSet(mock=[1, 2, 3, 4, 5])

    assert dice_set.get_dice(1).get() == 1
    assert dice_set.get_dice(2).get() == 2
    assert dice_set.get_dice(3).get() == 3
    assert dice_set.get_dice(4).get() == 4
    assert dice_set.get_dice(5).get() == 5


def test_dict_sort():
    dice_set = DiceSet()

    assert dice_set.get_as_array() == sorted(dice_set.get_as_array())

    mock_set = DiceSet(mock=[5, 4, 3, 2, 1])

    assert mock_set.get_as_array() == [1, 2, 3, 4, 5]

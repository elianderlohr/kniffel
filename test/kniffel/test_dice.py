import pytest

from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.dice import Dice


def test_random_throw():
    """Test 10 random dice throws and check if values are always between 1 and 6"""
    for _ in range(10):
        dice = Dice()
        assert dice.get() in [1, 2, 3, 4, 5, 6]

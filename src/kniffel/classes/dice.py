import random


class Dice:
    value = 0

    def __init__(self, mock: int = -1):
        """Dice class

        :param mock: list of mock dice values, defaults to -1
        :param logging: _description_, defaults to False
        """
        random.seed()

        if mock > 0:
            self.value = mock
        else:
            self.roll()

    def roll(self):
        """Roll the dice"""
        self.value = random.randint(1, 6)

    def get(self):
        """Get value of the dice

        :return: value of the dice
        """
        return self.value

    def set(self, value: int):
        """Set value of the dice

        :param value: value to set
        """
        self.value = value
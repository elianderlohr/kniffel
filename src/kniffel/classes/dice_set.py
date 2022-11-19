from src.kniffel.classes.dice import Dice


class DiceSet:
    dices = {}
    logging = False

    def __init__(self, mock: list = None, logging: bool = False):
        self.dices = {}

        if mock is None:
            self.roll()
        else:
            self.dices = {
                1: Dice(mock=mock[0]),
                2: Dice(mock=mock[1]),
                3: Dice(mock=mock[2]),
                4: Dice(mock=mock[3]),
                5: Dice(mock=mock[4]),
            }

        self.logging = logging

    def roll(self):
        """
        Roll the five dices
        """
        self.dices = {1: Dice(), 2: Dice(), 3: Dice(), 4: Dice(), 5: Dice()}

    def get(self):
        """
        Get dices
        """
        return self.dices

    def get_as_array(self):
        return [v.get() for _, v in self.dices.items()]

    def get_dice(self, index: int):
        """
        Get single dice by index

        :param int index: index of the dice
        """
        return self.dices[index]

    def set_dice(self, index: int, value: int):
        """
        Set the value of a dice in the set

        :param int index: index of dice to set value
        :param int value: value to set
        """
        self.dices[index] = value

    def to_list(self):
        """
        Transform list of dice objects to simple int array list
        """
        values = []
        for v in self.dices.values():
            values.append(v.get())
        return values

    def print(self):
        """
        Print dice array
        """
        print(self.to_list())

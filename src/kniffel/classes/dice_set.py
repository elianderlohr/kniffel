from src.kniffel.classes.dice import Dice


class DiceSet:
    dices: dict = {}
    logging: bool = False

    def __init__(
        self, mock: list = [], logging: bool = False, should_sort: bool = True
    ):
        """Initialize DiceSet

        :param mock: list, defaults to []
        :param logging: bool, defaults to False
        """
        self.dices: dict = {}

        self.should_sort = should_sort

        if len(mock) == 0:
            self.roll()
        else:
            if should_sort:
                mock = sorted(mock)

            self.dices[1] = Dice(mock=mock[0])
            self.dices[2] = Dice(mock=mock[1])
            self.dices[3] = Dice(mock=mock[2])
            self.dices[4] = Dice(mock=mock[3])
            self.dices[5] = Dice(mock=mock[4])

        self.logging = logging

    def roll(self):
        """
        Roll the five dices
        """
        self.dices[1] = Dice()
        self.dices[2] = Dice()
        self.dices[3] = Dice()
        self.dices[4] = Dice()
        self.dices[5] = Dice()

        self.sort()

    def sort(self):
        """
        Sort the dice set

        :param dict dice_set: dice set to sort
        """

        if self.should_sort:
            self.dices = dict(
                sorted(self.dices.items(), key=lambda item: item[1].value)
            )

            # Make sure the dices are sorted
            assert sorted(self.to_int_list()) == self.to_int_list()
        else:
            pass

    def get(self):
        """
        Get dices
        """
        return self.dices

    def to_dice_list(self):
        """
        Get dice values as list of Dice objects

        :return: list of Dice objects
        """
        return [v.get() for _, v in self.dices.items()]

    def get_dice(self, index: int):
        """
        Get single dice by index

        :param int index: index of the dice
        """
        return self.dices[index]

    def set_dice(self, index: int, dice: Dice):
        """
        Set the value of a dice in the set

        :param int index: index of dice to set value
        :param Dice dice: new dice to set
        """
        self.dices[index] = dice

        self.sort()

    def to_int_list(self):
        """
        Get dice values as list of integers

        :return: list of integers
        """
        return [v.get() for v in self.dices.values()]

    def print(self):
        """
        Print dice array
        """
        print(self.to_int_list())

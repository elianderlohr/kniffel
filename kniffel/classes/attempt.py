from classes.status import KniffelStatus
from classes.options import KniffelOptions
from classes.dice_set import DiceSet
from classes.kniffel_check import KniffelCheck
from classes.kniffel_option import KniffelOptionClass


class Attempt:
    attempts = []
    status = KniffelStatus.INIT
    option = KniffelOptions.DEFAULT
    selected_option: KniffelOptionClass = None
    logging = False

    def __init__(self, logging: bool = False):
        self.attempts = []
        self.logging = logging

    def is_active(self):
        """
        Is active attempt or finished attempt
        """
        if self.count() > 3:
            return False
        elif self.status is KniffelStatus.FINISHED:
            return False
        elif self.option is not KniffelOptions.DEFAULT:
            return False
        else:
            return True

    def count(self):
        """
        Get attempts count
        """
        return len(self.attempts)

    def add_attempt(self, keep: list = None, dice_set: DiceSet = None):
        """
        Add new attempt.
        Optionally keep selected dices

        :param list keep: hot encoded array which dices to keep. (1 = keep, 0 = re-roll)
        """
        if dice_set is None:
            dice_set = DiceSet()

        if self.count() >= 3:
            raise Exception("Cannot do more then 3 attempts per round.")
        else:
            if self.is_active() == True and self.count() > 0 and keep is not None:
                old_set = self.attempts[-1]

                counter = 1
                for i in range(len(keep)):
                    if keep[i] == 1:
                        dice_set.set_dice(
                            index=counter, value=old_set.get_dice(counter)
                        )
                    counter += 1

            self.attempts.append(dice_set)

    def finish_attempt(self, option: KniffelOptions):
        """
        Finish attempt

        :param KniffelOptions option: selected option how to finish the attempt
        """
        if self.is_active() is True:
            self.status = KniffelStatus.FINISHED
            self.option = option

            if option is KniffelOptions.ONES:
                self.selected_option = KniffelCheck().check_1(self.attempts[-1])
            if option is KniffelOptions.TWOS:
                self.selected_option = KniffelCheck().check_2(self.attempts[-1])
            if option is KniffelOptions.THREES:
                self.selected_option = KniffelCheck().check_3(self.attempts[-1])
            if option is KniffelOptions.FOURS:
                self.selected_option = KniffelCheck().check_4(self.attempts[-1])
            if option is KniffelOptions.FIVES:
                self.selected_option = KniffelCheck().check_5(self.attempts[-1])
            if option is KniffelOptions.SIXES:
                self.selected_option = KniffelCheck().check_6(self.attempts[-1])

            if option is KniffelOptions.THREE_TIMES:
                self.selected_option = KniffelCheck().check_three_times(
                    self.attempts[-1]
                )
            if option is KniffelOptions.FOUR_TIMES:
                self.selected_option = KniffelCheck().check_four_times(
                    self.attempts[-1]
                )
            if option is KniffelOptions.FULL_HOUSE:
                self.selected_option = KniffelCheck().check_full_house(
                    self.attempts[-1]
                )
            if option is KniffelOptions.SMALL_STREET:
                self.selected_option = KniffelCheck().check_small_street(
                    self.attempts[-1]
                )
            if option is KniffelOptions.LARGE_STREET:
                self.selected_option = KniffelCheck().check_large_street(
                    self.attempts[-1]
                )
            if option is KniffelOptions.KNIFFEL:
                self.selected_option = KniffelCheck().check_kniffel(self.attempts[-1])
            if option is KniffelOptions.CHANCE:
                self.selected_option = KniffelCheck().check_chance(self.attempts[-1])

    def get_latest(self):
        """
        Get latest attempt
        """
        return self.attempts[-1]

    def mock(self, mock: DiceSet):
        """
        Mock set of dices instead of random throws

        :param DiceSet mock: set of dices
        """
        self.add_attempt(dice_set=mock)

    def to_list(self):
        """
        Transform list of dice objects to simple int array list
        """
        values = []
        for v in self.attempts:
            values.append(v.to_list())
        return values

    def print(self):
        """
        Print attempts
        """
        if self.status is KniffelStatus.FINISHED:
            print(
                "Turn (finished): "
                + str(len(self.attempts))
                + " - "
                + str(self.to_list())
                + " - "
                + str(self.selected_option)
            )
        else:
            print("Turn: " + str(len(self.attempts)) + " - " + str(self.to_list()))

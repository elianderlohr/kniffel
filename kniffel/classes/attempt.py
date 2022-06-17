from select import select
from classes.status import KniffelStatus
from classes.options import KniffelOptions
from classes.dice_set import DiceSet
from classes.kniffel_check import KniffelCheck
from classes.kniffel_option import KniffelOptionClass
from enum import Enum


class Attempt:
    attempts = []
    status = KniffelStatus.INIT
    option = KniffelOptions.DEFAULT
    logging = False
    selected_option = None

    def __init__(self, logging: bool = False):
        self.attempts = []
        self.logging = logging

    def is_active(self):
        """
        Is active attempt or finished attempt
        """
        if self.count() > 3:
            return False
        elif self.status.value == KniffelStatus.FINISHED.value:
            return False
        elif self.option.value is not KniffelOptions.DEFAULT.value:
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
            if self.is_active() and self.count() > 0 and keep is not None:
                old_set = self.attempts[-1]

                counter = 1
                for i in range(len(keep)):
                    if keep[i] == 1:
                        dice_set.set_dice(
                            index=counter, value=old_set.get_dice(counter)
                        )
                    counter += 1

            self.attempts.append(dice_set)

    def finish_attempt(self, option: KniffelOptions, help: bool = False) -> KniffelOptionClass:
        """
        Finish attempt

        :param KniffelOptions option: selected option how to finish the attempt
        """
        if self.is_active():
            self.status = KniffelStatus.FINISHED
            self.option = option

            selected = None

            if option.value == KniffelOptions.ONES.value:
                selected = KniffelCheck().check_1(self.attempts[-1])
            elif option.value == KniffelOptions.TWOS.value:
                selected = KniffelCheck().check_2(self.attempts[-1])
            elif option.value == KniffelOptions.THREES.value:
                selected = KniffelCheck().check_3(self.attempts[-1])
            elif option.value == KniffelOptions.FOURS.value:
                selected = KniffelCheck().check_4(self.attempts[-1])
            elif option.value == KniffelOptions.FIVES.value:
                selected = KniffelCheck().check_5(self.attempts[-1])
            elif option.value == KniffelOptions.SIXES.value:
                selected = KniffelCheck().check_6(self.attempts[-1])

            elif option.value == KniffelOptions.THREE_TIMES.value:
                selected = KniffelCheck().check_three_times(self.attempts[-1])
            elif option.value == KniffelOptions.FOUR_TIMES.value:
                selected = KniffelCheck().check_four_times(self.attempts[-1])
            elif option.value == KniffelOptions.FULL_HOUSE.value:
                selected = KniffelCheck().check_full_house(self.attempts[-1])
            elif option.value == KniffelOptions.SMALL_STREET.value:
                selected = KniffelCheck().check_small_street(self.attempts[-1])
            elif option.value == KniffelOptions.LARGE_STREET.value:
                selected = KniffelCheck().check_large_street(self.attempts[-1])
            elif option.value == KniffelOptions.KNIFFEL.value:
                selected = KniffelCheck().check_kniffel(self.attempts[-1])
            elif option.value == KniffelOptions.CHANCE.value:
                selected = KniffelCheck().check_chance(self.attempts[-1])

            if help:
                if option.value == KniffelOptions.ONES_SLASH.value:
                    possibility = KniffelCheck().check_1(self.attempts[-1])
                    selected = KniffelOptionClass("ones_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.ONES_SLASH)
                elif option.value == KniffelOptions.TWOS_SLASH.value:
                    possibility = KniffelCheck().check_2(self.attempts[-1])
                    selected = KniffelOptionClass("twos_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.TWOS_SLASH)
                elif option.value == KniffelOptions.THREES_SLASH.value:
                    possibility = KniffelCheck().check_3(self.attempts[-1])
                    selected = KniffelOptionClass("threes_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.THREES_SLASH)
                elif option.value == KniffelOptions.FOURS_SLASH.value:
                    possibility = KniffelCheck().check_4(self.attempts[-1])
                    selected = KniffelOptionClass("fours_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.FOURS_SLASH)
                elif option.value == KniffelOptions.FIVES_SLASH.value:
                    possibility = KniffelCheck().check_5(self.attempts[-1])
                    selected = KniffelOptionClass("fives_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.FIVES_SLASH)
                elif option.value == KniffelOptions.SIXES_SLASH.value:
                    possibility = KniffelCheck().check_6(self.attempts[-1])
                    selected = KniffelOptionClass("sixes_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.SIXES_SLASH)
                
                
                elif option.value == KniffelOptions.THREE_TIMES_SLASH.value:
                    possibility = KniffelCheck().check_three_times(self.attempts[-1])
                    selected = KniffelOptionClass("three_times_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.THREE_TIMES_SLASH)
                elif option.value == KniffelOptions.FOUR_TIMES_SLASH.value:
                    possibility = KniffelCheck().check_four_times(self.attempts[-1])
                    selected = KniffelOptionClass("four_times_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.FOUR_TIMES_SLASH)
                elif option.value == KniffelOptions.FULL_HOUSE_SLASH.value:
                    possibility = KniffelCheck().check_full_house(self.attempts[-1])
                    selected = KniffelOptionClass("full_house_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.FULL_HOUSE_SLASH)
                elif option.value == KniffelOptions.SMALL_STREET_SLASH.value:
                    possibility = KniffelCheck().check_small_street(self.attempts[-1])
                    selected = KniffelOptionClass("small_street_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.SMALL_STREET_SLASH)
                elif option.value == KniffelOptions.LARGE_STREET_SLASH.value:
                    possibility = KniffelCheck().check_large_street(self.attempts[-1])
                    selected = KniffelOptionClass("large_street_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.LARGE_STREET_SLASH)
                elif option.value == KniffelOptions.KNIFFEL_SLASH.value:
                    possibility = KniffelCheck().check_kniffel(self.attempts[-1])
                    selected = KniffelOptionClass("kniffel_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.KNIFFEL_SLASH)
                elif option.value == KniffelOptions.CHANCE_SLASH.value:
                    possibility = KniffelCheck().check_chance(self.attempts[-1])
                    selected = KniffelOptionClass("chance_slash", possibility.points, ds=possibility.dice_set, id=KniffelOptions.CHANCE_SLASH)
            else:
                if option.value == KniffelOptions.ONES_SLASH.value:
                    possibility = KniffelCheck().check_1(self.attempts[-1])
                    selected = KniffelOptionClass("ones_slash", 0, ds=None, id=KniffelOptions.ONES_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.TWOS_SLASH.value:
                    possibility = KniffelCheck().check_2(self.attempts[-1])
                    selected = KniffelOptionClass("twos_slash", 0, ds=None, id=KniffelOptions.TWOS_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.THREES_SLASH.value:
                    possibility = KniffelCheck().check_3(self.attempts[-1])
                    selected = KniffelOptionClass("threes_slash", 0, ds=None, id=KniffelOptions.THREES_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.FOURS_SLASH.value:
                    possibility = KniffelCheck().check_4(self.attempts[-1])
                    selected = KniffelOptionClass("fours_slash", 0, ds=None, id=KniffelOptions.FOURS_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.FIVES_SLASH.value:
                    possibility = KniffelCheck().check_5(self.attempts[-1])
                    selected = KniffelOptionClass("fives_slash", 0, ds=None, id=KniffelOptions.FIVES_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.SIXES_SLASH.value:
                    possibility = KniffelCheck().check_6(self.attempts[-1])
                    selected = KniffelOptionClass("sixes_slash", 0, ds=None, id=KniffelOptions.SIXES_SLASH, is_possible=True if possibility.points == 0 else False)
                
                elif option.value == KniffelOptions.THREE_TIMES_SLASH.value:
                    possibility = KniffelCheck().check_three_times(self.attempts[-1])
                    selected = KniffelOptionClass("three_times_slash", 0, ds=None, id=KniffelOptions.THREE_TIMES_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.FOUR_TIMES_SLASH.value:
                    possibility = KniffelCheck().check_four_times(self.attempts[-1])
                    selected = KniffelOptionClass("four_times_slash", 0, ds=None, id=KniffelOptions.FOUR_TIMES_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.FULL_HOUSE_SLASH.value:
                    possibility = KniffelCheck().check_full_house(self.attempts[-1])
                    selected = KniffelOptionClass("full_house_slash", 0, ds=None, id=KniffelOptions.FULL_HOUSE_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.SMALL_STREET_SLASH.value:
                    possibility = KniffelCheck().check_small_street(self.attempts[-1])
                    selected = KniffelOptionClass("small_street_slash", 0, ds=None, id=KniffelOptions.SMALL_STREET_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.LARGE_STREET_SLASH.value:
                    possibility = KniffelCheck().check_large_street(self.attempts[-1])
                    selected = KniffelOptionClass("large_street_slash", 0, ds=None, id=KniffelOptions.LARGE_STREET_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.KNIFFEL_SLASH.value:
                    possibility = KniffelCheck().check_kniffel(self.attempts[-1])
                    selected = KniffelOptionClass("kniffel_slash", 0, ds=None, id=KniffelOptions.KNIFFEL_SLASH, is_possible=True if possibility.points == 0 else False)
                elif option.value == KniffelOptions.CHANCE_SLASH.value:
                    possibility = KniffelCheck().check_chance(self.attempts[-1])
                    selected = KniffelOptionClass("chance_slash", 0, ds=None, id=KniffelOptions.CHANCE_SLASH, is_possible=True if possibility.points == 0 else False)

            self.selected_option = selected

        return self.selected_option

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
        if self.status.value == KniffelStatus.FINISHED.value:
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

import os
import sys
import inspect
import tensorflow as tf

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from classes.status import KniffelStatus
from classes.options import KniffelOptions
from classes.dice_set import DiceSet
from classes.attempt import Attempt
from classes.kniffel_check import KniffelCheck
import classes.custom_exceptions as ex

import numpy as np
import itertools


class Kniffel:
    turns = []
    logging = False

    def __init__(self, logging: bool = False):
        self.turns = []
        self.logging = logging
        self.start()

    def get_selected_option(self, status: list, id: int) -> list:
        """Return the selected option from the attempt with the defined id

        :param status: list to append the info
        :param id: if of the attempt
        :return: return list with infos appended
        """
        if id < len(self.turns):
            selected_option = (
                0
                if self.turns[id].selected_option is None
                else self.turns[id].selected_option.get_id()
            )

            points = (
                0
                if self.turns[id].selected_option is None
                else self.turns[id].selected_option.points
            )
        else:
            selected_option = 0
            points = 0

        status.append(selected_option)
        status.append(points)
        return status

    def get_turn_as_array(self, id: int, with_option=True, only_last_two=False) -> list:
        """Get the turn with defined id as a list

        :param id: id of the attempt to return
        :param with_option: return selected option as well, defaults to True
        :param only_last_two: return only the last two attempts, defaults to False
        :return: list of attempts and optional the selected option
        """
        turn = []

        attempt1 = None
        attempt2 = None
        attempt3 = None
        selected_option = None

        if 0 <= id < len(self.turns):

            attempt1 = (
                np.array([0, 0, 0, 0, 0], dtype=np.int8)
                if len(self.turns[id].attempts) <= 0
                else np.array(self.turns[id].attempts[0].get_as_array(), dtype=np.int8)
            )

            attempt2 = (
                np.array([0, 0, 0, 0, 0], dtype=np.int8)
                if len(self.turns[id].attempts) <= 1
                else np.array(self.turns[id].attempts[1].get_as_array(), dtype=np.int8)
            )

            attempt3 = (
                np.array([0, 0, 0, 0, 0], dtype=np.int8)
                if len(self.turns[id].attempts) <= 2
                else np.array(self.turns[id].attempts[2].get_as_array(), dtype=np.int8)
            )

            if with_option:
                selected_option = (
                    np.array([0], dtype=np.int8)
                    if self.turns[id].selected_option is None
                    else np.array(
                        [self.turns[id].selected_option.get_id()], dtype=np.int8
                    )
                )
        else:
            attempt1 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
            attempt2 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
            attempt3 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
            selected_option = np.array([0], dtype=np.int8)

        if with_option:
            if only_last_two:
                if attempt3[0] == 0:
                    turn = list(itertools.chain(attempt1, attempt2, selected_option))
                else:
                    turn = list(itertools.chain(attempt2, attempt3, selected_option))
            else:
                turn = list(
                    itertools.chain(attempt1, attempt2, attempt3, selected_option)
                )
        else:
            if only_last_two:
                if attempt3[0] == 0:
                    turn = list(itertools.chain(attempt1, attempt2))
                else:
                    turn = list(itertools.chain(attempt2, attempt3))
            else:
                turn = list(itertools.chain(attempt1, attempt2, attempt3))

        return turn

    def get_state(self):
        """Get state of game as list of integers

        :return: state of game as list of integer
        """
        latest_turn_id = len(self.turns) - 1

        if self.turns[latest_turn_id].status == KniffelStatus.FINISHED:
            latest_turn_id += 1

        status = self.get_turn_as_array(
            latest_turn_id, with_option=False, only_last_two=False
        )

        self.get_selected_option(status, 0)
        self.get_selected_option(status, 1)
        self.get_selected_option(status, 2)
        self.get_selected_option(status, 3)
        self.get_selected_option(status, 4)
        self.get_selected_option(status, 5)
        self.get_selected_option(status, 6)
        self.get_selected_option(status, 7)
        self.get_selected_option(status, 8)
        self.get_selected_option(status, 9)
        self.get_selected_option(status, 10)
        self.get_selected_option(status, 11)
        self.get_selected_option(status, 12)

        return np.array([np.array(status)])

    def start(self):
        """
        Start a complete new game
        """
        self.add_turn()

    def add_turn(self, keep: list = None):
        """
        Add turn

        :param list keep: hot encoded list of which dice to keep (1 = keep, 0 = drop)
        """
        if self.turns_left() > 0:
            if self.is_new_game() or self.is_turn_finished():
                self.turns.append(Attempt())

            try:
                self.turns[-1].add_attempt(keep)
            except ex.Error as e:
                raise e
        else:
            raise ex.GameFinishedException(
                "Cannot play more then 13 rounds. Play a new game!"
            )

        if self.logging:
            print("Attempt:")
            print(f"   Keep: {keep}")
            print(f"   Array: {self.get_state()}")
            print(f"   Status: {self.status()}")

    def finish_turn(self, option: KniffelOptions) -> int:
        """
        Finish turn

        :param KniffelOptions option: selected option how to finish the turn
        """
        if self.is_option_possible(option):
            if (
                self.is_new_game() is False
                and self.is_turn_finished() is False
                and self.is_finished() is False
            ):
                kniffel_option = self.turns[-1].finish_attempt(option)

                if self.logging:
                    print("Finish:")
                    print(f"   Action: {option}")
                    print(f"   Array: {self.get_state()}")
                    print(f"   Status: {self.status()}")

                if self.turns_left() > 1:
                    self.add_turn()

                return kniffel_option.points
            elif self.is_finished():
                raise ex.GameFinishedException("Game finished!")
            elif self.is_new_game():
                raise ex.NewGameException("Cannot finish new game!")
            elif self.is_turn_finished():
                raise ex.TurnFinishedException("Cannot finish finished round!")
        else:
            raise ex.SelectedOptionException(
                "Cannot select the same Option again or not possible for this. Select another Option!"
            )

    def get_points(self):
        """
        Get the total points
        """
        total = 0
        if self.turns is not [] and self.turns is not None:
            for turn in self.turns:
                if (
                    turn.status.value == KniffelStatus.FINISHED.value
                    and turn.selected_option is not None
                ):
                    total += turn.selected_option.points

            if self.is_bonus() is True:
                total += 35

        return total

    def is_option_possible(self, option: KniffelOptions):
        """
        Is Option possible

        :param KniffelOptions option: kniffel option to check
        """
        check = self.system_check()
        if option.value in check.keys():
            if check[option.value]:
                for turn in self.turns:
                    if option.value <= 13:
                        if option.value == turn.option.value:
                            return False
                        if option.value + 13 == turn.option.value:
                            return False
                    else:
                        if option.value - 13 == turn.option.value:
                            return False
                        if option.value == turn.option.value:
                            return False

                return True

        return False

    def is_bonus(self):
        """
        Is bonus possible.
        Sum for einser, zweier, dreier, vierer, fünfer und sechser needs to be higher or equal to 63
        """
        total = 0
        for turn in self.turns:
            if turn.status.value == KniffelStatus.FINISHED.value and (
                turn.option.value == KniffelOptions.ONES.value
                or turn.option.value == KniffelOptions.TWOS.value
                or turn.option.value == KniffelOptions.THREES.value
                or turn.option.value == KniffelOptions.FOURS.value
                or turn.option.value == KniffelOptions.FIVES.value
                or turn.option.value == KniffelOptions.SIXES.value
            ):
                total += turn.selected_option.points

        return True if total >= 63 else False

    def get_played_rounds(self):
        """Count the number of round

        :return: number of rounds played
        """
        rounds = 0
        for turn in self.turns:
            rounds += len(turn.attempts)

        return rounds

    def turns_left(self):
        """
        How many turns are left
        """
        return 14 - len(self.turns)

    def is_new_game(self):
        """
        Is the game new
        """
        if len(self.turns) == 0:
            return True
        else:
            return False

    def is_turn_finished(self):
        """
        Is the turn finished
        """
        if self.is_new_game() is False:
            if self.turns[-1].status.value == KniffelStatus.FINISHED.value:
                return True
            else:
                return False
        else:
            return True

    def is_finished(self):
        if self.turns_left() == 1:
            return True
        else:
            return False

    def system_check(self):
        """
        Check latest dice set for possible points
        """
        latest_turn = self.turns[-1]

        ds = latest_turn.attempts[-1]

        check = dict()

        check[KniffelOptions.ONES.value] = (
            True if KniffelCheck().check_1(ds).is_possible else False
        )
        check[KniffelOptions.TWOS.value] = (
            True if KniffelCheck().check_2(ds).is_possible else False
        )
        check[KniffelOptions.THREES.value] = (
            True if KniffelCheck().check_3(ds).is_possible else False
        )
        check[KniffelOptions.FOURS.value] = (
            True if KniffelCheck().check_4(ds).is_possible else False
        )
        check[KniffelOptions.FIVES.value] = (
            True if KniffelCheck().check_5(ds).is_possible else False
        )
        check[KniffelOptions.SIXES.value] = (
            True if KniffelCheck().check_6(ds).is_possible else False
        )
        check[KniffelOptions.THREE_TIMES.value] = (
            True if KniffelCheck().check_three_times(ds).is_possible else False
        )
        check[KniffelOptions.FOUR_TIMES.value] = (
            True if KniffelCheck().check_four_times(ds).is_possible else False
        )
        check[KniffelOptions.FULL_HOUSE.value] = (
            True if KniffelCheck().check_full_house(ds).is_possible else False
        )
        check[KniffelOptions.SMALL_STREET.value] = (
            True if KniffelCheck().check_small_street(ds).is_possible else False
        )
        check[KniffelOptions.LARGE_STREET.value] = (
            True if KniffelCheck().check_large_street(ds).is_possible else False
        )
        check[KniffelOptions.KNIFFEL.value] = (
            True if KniffelCheck().check_kniffel(ds).is_possible else False
        )
        check[KniffelOptions.CHANCE.value] = (
            True if KniffelCheck().check_chance(ds).is_possible else False
        )

        check[KniffelOptions.ONES_SLASH.value] = (
            False if KniffelCheck().check_1(ds).is_possible else True
        )
        check[KniffelOptions.TWOS_SLASH.value] = (
            False if KniffelCheck().check_2(ds).is_possible else True
        )
        check[KniffelOptions.THREES_SLASH.value] = (
            False if KniffelCheck().check_3(ds).is_possible else True
        )
        check[KniffelOptions.FOURS_SLASH.value] = (
            False if KniffelCheck().check_4(ds).is_possible else True
        )
        check[KniffelOptions.FIVES_SLASH.value] = (
            False if KniffelCheck().check_5(ds).is_possible else True
        )
        check[KniffelOptions.SIXES_SLASH.value] = (
            False if KniffelCheck().check_6(ds).is_possible else True
        )
        check[KniffelOptions.THREE_TIMES_SLASH.value] = (
            False if KniffelCheck().check_three_times(ds).is_possible else True
        )
        check[KniffelOptions.FOUR_TIMES_SLASH.value] = (
            False if KniffelCheck().check_four_times(ds).is_possible else True
        )
        check[KniffelOptions.FULL_HOUSE_SLASH.value] = (
            False if KniffelCheck().check_full_house(ds).is_possible else True
        )
        check[KniffelOptions.SMALL_STREET_SLASH.value] = (
            False if KniffelCheck().check_small_street(ds).is_possible else True
        )
        check[KniffelOptions.LARGE_STREET_SLASH.value] = (
            False if KniffelCheck().check_large_street(ds).is_possible else True
        )
        check[KniffelOptions.KNIFFEL_SLASH.value] = (
            False if KniffelCheck().check_kniffel(ds).is_possible else True
        )
        check[KniffelOptions.CHANCE_SLASH.value] = (
            False if KniffelCheck().check_chance(ds).is_possible else True
        )

        return check

    def mock(self, mock: DiceSet):
        """
        Mock turn play

        :param DiceSet mock: mock dice set
        """
        if self.is_new_game() is True or self.is_turn_finished() is True:
            self.turns.append(Attempt())

        self.turns[-1].mock(mock)

    def print_check(self):
        """
        Print the check of possible options
        """
        options = {
            k: v for k, v in self.system_check().items() if v.is_possible == True
        }
        print(options)

    def to_list(self):
        """
        Transform list of dice objects to simple int array list
        """
        values = []
        for v in self.turns:
            values.append(v.to_list())
        return values

    def print(self):
        """
        Print list
        """
        i = 1
        for round in self.turns:
            print("# Turn: " + str(i) + "/13")
            round.print()
            i += 1

    def status(self):
        status = {}
        status["game-over"] = self.is_finished()
        status["bonus"] = self.is_bonus()
        return status
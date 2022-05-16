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

import numpy as np
import itertools


class Kniffel:
    turns = []
    logging = False

    def __init__(self, logging: bool = False):
        self.turns = []
        self.logging = logging

        self.add_turn()

    def get_turn_as_dict(self, id: int):
        if 0 <= id < len(self.turns):
            turn = {
                "attempt1": [0, 0, 0, 0, 0]
                if len(self.turns[id].attempts) <= 0
                else self.turns[id].attempts[0].get_as_array(),
                "attempt2": [0, 0, 0, 0, 0]
                if len(self.turns[id].attempts) <= 1
                else self.turns[id].attempts[1].get_as_array(),
                "attempt3": [0, 0, 0, 0, 0]
                if len(self.turns[id].attempts) <= 2
                else self.turns[id].attempts[2].get_as_array(),
                "selected_option": 0
                if self.turns[id].selected_option is None
                else self.turns[id].selected_option.get_id(),
            }
        else:
            turn = {
                "attempt1": [0, 0, 0, 0, 0],
                "attempt2": [0, 0, 0, 0, 0],
                "attempt3": [0, 0, 0, 0, 0],
                "selected_option": -1,
            }

        return turn

    def get_turn_as_array(self, id: int):
        if 0 <= id < len(self.turns):
            turn = []
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

            selected_option = (
                np.array([0], dtype=np.int8)
                if self.turns[id].selected_option is None
                else np.array([self.turns[id].selected_option.get_id()], dtype=np.int8)
            )

            turn = list(itertools.chain(attempt1, attempt2, attempt3, selected_option))
        else:
            attempt1 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
            attempt2 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
            attempt3 = np.array([0, 0, 0, 0, 0], dtype=np.int8)
            selected_option = np.array([0], dtype=np.int8)

            turn = list(itertools.chain(attempt1, attempt2, attempt3, selected_option))

        return turn

    def get_dict(self):
        turns = {
            "turn1": self.get_turn_as_dict(0),
            "turn2": self.get_turn_as_dict(1),
            "turn3": self.get_turn_as_dict(2),
            "turn4": self.get_turn_as_dict(3),
            "turn5": self.get_turn_as_dict(4),
            "turn6": self.get_turn_as_dict(5),
            "turn7": self.get_turn_as_dict(6),
            "turn8": self.get_turn_as_dict(7),
            "turn9": self.get_turn_as_dict(8),
            "turn10": self.get_turn_as_dict(9),
            "turn11": self.get_turn_as_dict(10),
            "turn12": self.get_turn_as_dict(11),
            "turn13": self.get_turn_as_dict(12),
        }

        return turns

    def get_array(self):
        turns = []

        turns.append(self.get_turn_as_array(0))
        turns.append(self.get_turn_as_array(1))
        turns.append(self.get_turn_as_array(2))
        turns.append(self.get_turn_as_array(3))
        turns.append(self.get_turn_as_array(4))
        turns.append(self.get_turn_as_array(5))
        turns.append(self.get_turn_as_array(6))
        turns.append(self.get_turn_as_array(7))
        turns.append(self.get_turn_as_array(8))
        turns.append(self.get_turn_as_array(9))
        turns.append(self.get_turn_as_array(10))
        turns.append(self.get_turn_as_array(11))
        turns.append(self.get_turn_as_array(12))

        # return tf.convert_to_tensor(np.array(turns, dtype=np.int8), dtype=tf.int8)
        return np.asarray(turns, dtype=np.int32).reshape(13, 16)

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

            self.turns[-1].add_attempt(keep)
        else:
            raise Exception("Cannot play more then 13 rounds. Play a new game!")

    def finish_turn(self, option: KniffelOptions) -> int:
        """
        Finish turn

        :param KniffelOptions option: selected option how to finish the turn
        """
        if self.is_option_possible(option):
            if self.is_new_game() is False and self.is_turn_finished() is False:
                kniffel_option = self.turns[-1].finish_attempt(option)
                return kniffel_option.points

            elif self.is_new_game():
                raise Exception("Cannot finish new game!")
            elif self.is_turn_finished():
                raise Exception("Cannot finish finished round!")
        else:
            raise Exception(
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
            if check[option.value].is_possible:
                for turn in self.turns:
                    if turn.option is option:
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
        return 13 - len(self.turns)

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
        if self.turns_left == 0:
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

        check[KniffelOptions.ONES.value] = KniffelCheck().check_1(ds)
        check[KniffelOptions.TWOS.value] = KniffelCheck().check_2(ds)
        check[KniffelOptions.THREES.value] = KniffelCheck().check_3(ds)
        check[KniffelOptions.FOURS.value] = KniffelCheck().check_4(ds)
        check[KniffelOptions.FIVES.value] = KniffelCheck().check_5(ds)
        check[KniffelOptions.SIXES.value] = KniffelCheck().check_6(ds)
        check[KniffelOptions.THREE_TIMES.value] = KniffelCheck().check_three_times(ds)
        check[KniffelOptions.FOUR_TIMES.value] = KniffelCheck().check_four_times(ds)
        check[KniffelOptions.FULL_HOUSE.value] = KniffelCheck().check_full_house(ds)
        check[KniffelOptions.SMALL_STREET.value] = KniffelCheck().check_small_street(ds)
        check[KniffelOptions.LARGE_STREET.value] = KniffelCheck().check_large_street(ds)
        check[KniffelOptions.KNIFFEL.value] = KniffelCheck().check_kniffel(ds)
        check[KniffelOptions.CHANCE.value] = KniffelCheck().check_chance(ds)

        return check

    def check(self):
        """
        Check latest dice set for possible points
        """
        latest_turn = self.turns[-1]

        ds = latest_turn.attempts[-1]

        check = dict()

        check["ones"] = KniffelCheck().check_1(ds)
        check["twos"] = KniffelCheck().check_2(ds)
        check["threes"] = KniffelCheck().check_3(ds)
        check["fours"] = KniffelCheck().check_4(ds)
        check["fives"] = KniffelCheck().check_5(ds)
        check["sixes"] = KniffelCheck().check_6(ds)
        check["three-time"] = KniffelCheck().check_three_times(ds)
        check["four-time"] = KniffelCheck().check_four_times(ds)
        check["full-house"] = KniffelCheck().check_full_house(ds)
        check["small-street"] = KniffelCheck().check_small_street(ds)
        check["large-street"] = KniffelCheck().check_large_street(ds)
        check["kniffel"] = KniffelCheck().check_kniffel(ds)
        check["chance"] = KniffelCheck().check_chance(ds)

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
        options = {k: v for k, v in self.check().items() if v.is_possible == True}
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

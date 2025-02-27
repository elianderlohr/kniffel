import os
import sys
import inspect
import tensorflow as tf
import numpy as np
import itertools

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from src.kniffel.classes.status import KniffelStatus
from src.kniffel.classes.options import KniffelOptions
from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.attempt import Attempt
from src.kniffel.classes.kniffel_check import KniffelCheck
import src.kniffel.classes.custom_exceptions as ex
from src.kniffel.classes.kniffel_option import KniffelOptionClass


class Kniffel:
    """Kniffel game class

    Raises:
        e: error
        ex.GameFinishedException: Exception when the game is properly finished
        ex.NewGameException: Exception when the game is not started
        ex.TurnFinishedException: Exception with the turns
        ex.SelectedOptionException: Exception when wrong option is selected
    """

    turns: list = []
    logging = False

    def __init__(
        self, logging: bool = False, custom=False, state_mode="binary"
    ):  # state_mode: binary, continuous
        self.turns: list = []
        self.logging = logging
        self.state_mode = state_mode

        if not custom:
            self.start()

    def get_length(self) -> int:
        return len(self.turns)

    def get(self, id: int) -> Attempt:
        if id < self.get_length():
            return self.turns[id]
        else:
            raise Exception("Attempt not found")

    def get_selected_option(self, status: list, id: int) -> list:
        """Return the selected option from the attempt with the defined id

        :param status: list to append the info
        :param id: if of the attempt
        :return: return list with infos appended
        """
        if id < self.get_length():
            selected_option = (
                0
                if self.get(id).selected_option is None
                else self.get(id).selected_option.get_id()
            )

            points = (
                0
                if self.get(id).selected_option is None
                else self.get(id).selected_option.get_points()
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
        selected_option: list = []

        if 0 <= id < self.get_length():
            attempt1 = (
                np.array([0, 0, 0, 0, 0], dtype=np.int8)
                if len(self.get(id).attempts) <= 0
                else np.array(self.get(id).get_attempt(0).to_dice_list(), dtype=np.int8)
            )

            attempt2 = (
                np.array([0, 0, 0, 0, 0], dtype=np.int8)
                if len(self.get(id).attempts) <= 1
                else np.array(self.get(id).get_attempt(1).to_dice_list(), dtype=np.int8)
            )

            attempt3 = (
                np.array([0, 0, 0, 0, 0], dtype=np.int8)
                if len(self.get(id).attempts) <= 2
                else np.array(self.get(id).get_attempt(2).to_dice_list(), dtype=np.int8)
            )

            if with_option:
                selected_option = (
                    np.array([0], dtype=np.int8)
                    if self.get(id).get_selected_option() is None
                    else np.array(
                        [self.get(id).get_selected_option().get_id()], dtype=np.int8
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

    def get_last_id(self) -> int:
        latest_turn_id = len(self.turns) - 1

        if self.get_turn(latest_turn_id).status == KniffelStatus.FINISHED:
            latest_turn_id += 1

        return latest_turn_id

    def get_last(self) -> Attempt:
        return self.get_turn(-1)

    def get_turn(self, id: int) -> Attempt:
        return self.turns[id]

    def get_option_point(
        self,
        option: KniffelOptions,
        option_alternative: KniffelOptions,
        scaler: int,
    ) -> float:
        for id in range(self.get_length()):
            turn = self.get_turn(id)

            if turn.status is KniffelStatus.FINISHED:
                if turn.get_selected_option().id is option.value:
                    if self.state_mode == "continuous":
                        return turn.get_selected_option().points / scaler
                    else:
                        return 1
                elif turn.get_selected_option().id is option_alternative.value:
                    return 0

        return -1

    def get_option_kniffel_points(
        self, option: KniffelOptions, option_alternative: KniffelOptions
    ):
        for id in range(self.get_length()):
            turn = self.get_turn(id)

            if turn.status is KniffelStatus.FINISHED:
                if turn.get_selected_option().id is option.value:
                    return turn.get_selected_option().points
                elif turn.get_selected_option().id is option_alternative.value:
                    return -10

        return 0

    def create_one_hot_dice_encoding(self, status: list, dice: int):
        for i in range(1, 6 + 1):  # loop from 1 to 6 inclusive
            if i == dice:
                status.append(1)
            else:
                status.append(0)

        return status

    def get_state(self):
        """Get state of game as list of integers

        :return: state of game as list of integer
        """
        turn = self.get_last()

        status = list()
        status = self.create_one_hot_dice_encoding(
            status, turn.get_latest().to_int_list()[0]
        )
        status = self.create_one_hot_dice_encoding(
            status, turn.get_latest().to_int_list()[1]
        )
        status = self.create_one_hot_dice_encoding(
            status, turn.get_latest().to_int_list()[2]
        )
        status = self.create_one_hot_dice_encoding(
            status, turn.get_latest().to_int_list()[3]
        )
        status = self.create_one_hot_dice_encoding(
            status, turn.get_latest().to_int_list()[4]
        )

        # Tries played
        if self.get_last().count() == 1:
            status.append(1)
            status.append(0)
            status.append(0)

        if self.get_last().count() == 2:
            status.append(0)
            status.append(1)
            status.append(0)

        if self.get_last().count() == 3:
            status.append(0)
            status.append(0)
            status.append(1)

        # Bonus ?
        status.append(1 if self.is_bonus() else 0)

        status.append(
            self.get_option_point(
                KniffelOptions.ONES,
                KniffelOptions.ONES_SLASH,
                5,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.TWOS,
                KniffelOptions.TWOS_SLASH,
                10,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.THREES,
                KniffelOptions.THREES_SLASH,
                15,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.FOURS,
                KniffelOptions.FOURS_SLASH,
                20,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.FIVES,
                KniffelOptions.FIVES_SLASH,
                25,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.SIXES,
                KniffelOptions.SIXES_SLASH,
                30,
            )
        )

        status.append(
            self.get_option_point(
                KniffelOptions.THREE_TIMES,
                KniffelOptions.THREE_TIMES_SLASH,
                30,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.FOUR_TIMES,
                KniffelOptions.FOUR_TIMES_SLASH,
                30,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.FULL_HOUSE,
                KniffelOptions.FULL_HOUSE_SLASH,
                25,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.SMALL_STREET,
                KniffelOptions.SMALL_STREET_SLASH,
                30,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.LARGE_STREET,
                KniffelOptions.LARGE_STREET_SLASH,
                40,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.KNIFFEL,
                KniffelOptions.KNIFFEL_SLASH,
                50,
            )
        )
        status.append(
            self.get_option_point(
                KniffelOptions.CHANCE,
                KniffelOptions.CHANCE_SLASH,
                30,
            )
        )

        return np.array([np.array(status, dtype=np.float16)], dtype=np.float16)
        # return np.array([np.array(status)])

    def start(self):
        """
        Start a complete new game
        """
        self.add_turn()

    def add_turn(self, keep: list = []):
        """
        Add turn

        :param list keep: hot encoded list of which dice to keep (1 = keep, 0 = drop)
        """
        if self.turns_left() > 0:
            if self.is_new_game() or self.is_turn_finished():
                self.turns.append(Attempt())
            try:
                self.get_last().add_attempt(keep)
            except Exception as e:
                raise e
        else:
            raise ex.GameFinishedException()

    def get_alternative_action(self, option: KniffelOptions):
        """Get alternative action
        :param option: option
        :return: alternative action
        """
        if option == KniffelOptions.ONES:
            return KniffelOptions.ONES_SLASH
        elif option == KniffelOptions.TWOS:
            return KniffelOptions.TWOS_SLASH
        elif option == KniffelOptions.THREES:
            return KniffelOptions.THREES_SLASH
        elif option == KniffelOptions.FOURS:
            return KniffelOptions.FOURS_SLASH
        elif option == KniffelOptions.FIVES:
            return KniffelOptions.FIVES_SLASH
        elif option == KniffelOptions.SIXES:
            return KniffelOptions.SIXES_SLASH
        elif option == KniffelOptions.THREE_TIMES:
            return KniffelOptions.THREE_TIMES_SLASH
        elif option == KniffelOptions.FOUR_TIMES:
            return KniffelOptions.FOUR_TIMES_SLASH
        elif option == KniffelOptions.FULL_HOUSE:
            return KniffelOptions.FULL_HOUSE_SLASH
        elif option == KniffelOptions.SMALL_STREET:
            return KniffelOptions.SMALL_STREET_SLASH
        elif option == KniffelOptions.LARGE_STREET:
            return KniffelOptions.LARGE_STREET_SLASH
        elif option == KniffelOptions.KNIFFEL:
            return KniffelOptions.KNIFFEL_SLASH
        elif option == KniffelOptions.CHANCE:
            return KniffelOptions.CHANCE_SLASH
        else:
            return option

    def finish_turn(self, option: KniffelOptions) -> KniffelOptionClass:
        """
        Finish turn

        :param KniffelOptions option: selected option how to finish the turn
        """
        if self.is_option_possible(option):
            if self.is_new_game() is False and self.is_turn_finished() is False:
                kniffel_option = self.get_last().finish_attempt(option)

                if self.is_finished():
                    raise ex.GameFinishedException()
                elif self.turns_left() > 1:
                    self.add_turn()

                return kniffel_option

            elif self.is_new_game():
                raise ex.NewGameException()
            elif self.is_turn_finished():
                raise ex.TurnFinishedException()
        else:
            raise ex.SelectedOptionException()

        return None  # type: ignore

    def get_points(self):
        """
        Get the total points
        """
        total = 0
        if self.turns is not [] and self.turns is not None:
            for turn in self.turns:
                if (
                    turn.status is KniffelStatus.FINISHED
                    and turn.selected_option is not None
                ):
                    total += turn.selected_option.points

            if self.is_bonus():  #  and self.is_finished():
                total += 35

        return total

    def get_points_top(self) -> int:
        """Sum for einser, zweier, dreier, vierer, fünfer und sechser

        Returns:
            int: total points top with bonus
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

        total += 35 if self.is_bonus() else 0

        return total

    def get_points_bottom(self):
        total = 0
        for turn in self.turns:
            if turn.status.value == KniffelStatus.FINISHED.value and (
                turn.option.value == KniffelOptions.THREE_TIMES.value
                or turn.option.value == KniffelOptions.FOUR_TIMES.value
                or turn.option.value == KniffelOptions.FULL_HOUSE.value
                or turn.option.value == KniffelOptions.SMALL_STREET.value
                or turn.option.value == KniffelOptions.LARGE_STREET.value
                or turn.option.value == KniffelOptions.KNIFFEL.value
                or turn.option.value == KniffelOptions.CHANCE.value
            ):
                total += turn.selected_option.points

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
                    # check if option or alternative option is already selected
                    if (
                        turn.option.value == option.value
                        or turn.option.value
                        == self.get_alternative_action(option).value
                    ):
                        return False

                # if option is not already selected, return true
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
        if self.turns_left() == 1 and self.get_last().status is KniffelStatus.FINISHED:
            return True
        else:
            return False

    def system_check(self):
        """
        Check latest dice set for possible points
        """
        latest_turn = self.get_last()

        ds = latest_turn.get_latest()

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

        check[KniffelOptions.ONES_SLASH.value] = True
        check[KniffelOptions.TWOS_SLASH.value] = True
        check[KniffelOptions.THREES_SLASH.value] = True
        check[KniffelOptions.FOURS_SLASH.value] = True
        check[KniffelOptions.FIVES_SLASH.value] = True
        check[KniffelOptions.SIXES_SLASH.value] = True
        check[KniffelOptions.THREE_TIMES_SLASH.value] = True
        check[KniffelOptions.FOUR_TIMES_SLASH.value] = True
        check[KniffelOptions.FULL_HOUSE_SLASH.value] = True
        check[KniffelOptions.SMALL_STREET_SLASH.value] = True
        check[KniffelOptions.LARGE_STREET_SLASH.value] = True
        check[KniffelOptions.KNIFFEL_SLASH.value] = True
        check[KniffelOptions.CHANCE_SLASH.value] = True

        return check

    def mock(self, mock: DiceSet):
        """
        Mock turn play

        :param DiceSet mock: mock dice set
        """
        if self.is_new_game() or self.is_turn_finished():
            self.turns.append(Attempt())

        self.turns[-1].mock(mock)

    def print_check(self):
        """
        Print the check of possible options
        """
        options = {k: v for k, v in self.system_check().items() if v.is_possible}
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

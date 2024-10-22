from enum import Enum
import logging
import os
import sys
import inspect
import csv
import math

from sqlalchemy import false

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))  # type: ignore
    ),
)

from src.kniffel.classes.options import KniffelOptions
from src.kniffel.classes.kniffel import Kniffel


class EnumAction(Enum):
    # Finish Actions
    FINISH_ONES = 0
    FINISH_TWOS = 1
    FINISH_THREES = 2
    FINISH_FOURS = 3
    FINISH_FIVES = 4
    FINISH_SIXES = 5
    FINISH_THREE_TIMES = 6
    FINISH_FOUR_TIMES = 7
    FINISH_FULL_HOUSE = 8
    FINISH_SMALL_STREET = 9
    FINISH_LARGE_STREET = 10
    FINISH_KNIFFEL = 11
    FINISH_CHANCE = 12

    # Continue Actions
    NEXT_0 = 13  # 0	    0	0	0	0	0
    NEXT_1 = 14  # 1	    0	0	0	0	1
    NEXT_2 = 15  # 2	    0	0	0	1	0
    NEXT_3 = 16  # 3	    0	0	0	1	1
    NEXT_4 = 17  # 4	    0	0	1	0	0
    NEXT_5 = 18  # 5	    0	0	1	0	1
    NEXT_6 = 19  # 6	    0	0	1	1	0
    NEXT_7 = 20  # 7	    0	0	1	1	1
    NEXT_8 = 21  # 8	    0	1	0	0	0
    NEXT_9 = 22  # 9	    0	1	0	0	1
    NEXT_10 = 23  # 10	    0	1	0	1	0
    NEXT_11 = 24  # 11	    0	1	0	1	1
    NEXT_12 = 25  # 12	    0	1	1	0	0
    NEXT_13 = 26  # 13	    0	1	1	0	1
    NEXT_14 = 27  # 14	    0	1	1	1	0
    NEXT_15 = 28  # 15	    0	1	1	1	1
    NEXT_16 = 29  # 16	    1	0	0	0	0
    NEXT_17 = 30  # 17	    1	0	0	0	1
    NEXT_18 = 31  # 18	    1	0	0	1	0
    NEXT_19 = 32  # 19	    1	0	0	1	1
    NEXT_20 = 33  # 20	    1	0	1	0	0
    NEXT_21 = 34  # 21	    1	0	1	0	1
    NEXT_22 = 35  # 22	    1	0	1	1	0
    NEXT_23 = 36  # 23	    1	0	1	1	1
    NEXT_24 = 37  # 24  	1	1	0	0	0
    NEXT_25 = 38  # 25  	1	1	0	0	1
    NEXT_26 = 39  # 26  	1	1	0	1	0
    NEXT_27 = 40  # 27  	1	1	0	1	1
    NEXT_28 = 41  # 28  	1	1	1	0	0
    NEXT_29 = 42  # 29  	1	1	1	0	1
    NEXT_30 = 43  # 30  	1	1	1	1	0

    # Finish Actions
    FINISH_ONES_SLASH = 44
    FINISH_TWOS_SLASH = 45
    FINISH_THREES_SLASH = 46
    FINISH_FOURS_SLASH = 47
    FINISH_FIVES_SLASH = 48
    FINISH_SIXES_SLASH = 49
    FINISH_THREE_TIMES_SLASH = 50
    FINISH_FOUR_TIMES_SLASH = 51
    FINISH_FULL_HOUSE_SLASH = 52
    FINISH_SMALL_STREET_SLASH = 53
    FINISH_LARGE_STREET_SLASH = 54
    FINISH_KNIFFEL_SLASH = 55
    FINISH_CHANCE_SLASH = 56

    FINISH_GAME = 99


class KniffelConfig(Enum):
    """
    Kniffel Option Enum
    """

    ONES = "reward_ones"
    TWOS = "reward_twos"
    THREES = "reward_threes"
    FOURS = "reward_fours"
    FIVES = "reward_fives"
    SIXES = "reward_sixes"
    THREE_TIMES = "reward_three_times"
    FOUR_TIMES = "reward_four_times"
    FULL_HOUSE = "reward_full_house"
    SMALL_STREET = "reward_small_street"
    LARGE_STREET = "reward_large_street"
    KNIFFEL = "reward_kniffel"
    CHANCE = "reward_chance"

    FIVE_DICES = "reward_five_dices"
    FOUR_DICES = "reward_four_dices"
    THREE_DICES = "reward_three_dices"
    TWO_DICES = "reward_two_dices"
    ONE_DICES = "reward_one_dice"
    SLASH = "reward_slash"


class KniffelEnvHelper:
    # kniffel instance
    kniffel: Kniffel

    # env_config
    config: dict

    def __init__(
        self,
        env_config,
        logging=False,
        custom_kniffel=False,
        reward_mode="kniffel",  # kniffel, custom
        state_mode="binary",  # binary, continuous
    ):
        # http://www.brefeld.homepage.t-online.de/kniffel.html
        # http://www.brefeld.homepage.t-online.de/kniffel-strategie.html

        self.kniffel = Kniffel(
            logging=logging, custom=custom_kniffel, state_mode=state_mode
        )

        self.logging = logging

        self.custom_kniffel = custom_kniffel

        self.reward_mode = reward_mode
        self.state_mode = state_mode

        self.config = env_config["reward_kniffel"]

        if "reward_roll_dice" in env_config:
            self._reward_roll_dice = env_config["reward_roll_dice"]
        else:
            self._reward_roll_dice = 0
            print("reward_roll_dice not in env_config, set to default '0'")

        if "reward_game_over" in env_config:
            self._reward_game_over = env_config["reward_game_over"]
        else:
            self._reward_game_over = 25
            print("reward_game_over not in env_config, set to default '25'")

        if "reward_bonus" in env_config:
            self._reward_bonus = env_config["reward_bonus"]
        else:
            self._reward_bonus = 25
            print("reward_bonus not in env_config, set to default '25'")

        if "reward_finish" in env_config:
            self._reward_finish = env_config["reward_finish"]
        else:
            self._reward_finish = 5
            print("reward_finish not in env_config, set to default '5'")

    def reset_kniffel(self):
        del self.kniffel
        self.kniffel = Kniffel(
            logging=self.logging, custom=self.custom_kniffel, state_mode=self.state_mode
        )

    def get_state(self):
        return self.kniffel.get_state()

    def get_config_param(
        self, option: KniffelConfig, dice_count: KniffelConfig
    ) -> float:
        """Return config

        Args:
            option (KniffelConfig): Kniffel option
            dice_count (KniffelConfig): Kniffel dice amount

        Returns:
            float: Reward float
        """
        return float(self.config[option.value][dice_count.value])

    def put_parameter(self, parameters: dict, key: str, alternative: float) -> float:
        """Take passed parameter or take alternative

        :param parameters: dict of parameters
        :param key: key of the parameter
        :param alternative: default value for the parameter
        :return: value of the parameter
        """
        if key in parameters.keys():
            return float(parameters[key])
        else:
            return alternative

    def count_same_dice(self, dice: list) -> int:
        return max(dice.count(i) for i in dice)

    def rewards_single(self, kniffel_type: KniffelConfig, dice_count: int) -> float:
        """Calculate reward based on amount of dices used for finishing the round.

        :param dice_count: amount of dices
        :return: reward
        """
        if dice_count == 0:
            return self.get_config_param(kniffel_type, KniffelConfig.SLASH)
        elif dice_count == 1:
            return self.get_config_param(kniffel_type, KniffelConfig.ONE_DICES)
        elif dice_count == 2:
            return self.get_config_param(kniffel_type, KniffelConfig.TWO_DICES)
        elif dice_count == 3:
            return self.get_config_param(kniffel_type, KniffelConfig.THREE_DICES)
        elif dice_count == 4:
            return self.get_config_param(kniffel_type, KniffelConfig.FOUR_DICES)
        else:
            return self.get_config_param(kniffel_type, KniffelConfig.FIVE_DICES)

    def reward_chance(self, score) -> float:
        reward = 0
        if score >= 5:
            reward = self.get_config_param(
                KniffelConfig.CHANCE, KniffelConfig.ONE_DICES
            )
        if score >= 10:
            reward = self.get_config_param(
                KniffelConfig.CHANCE, KniffelConfig.TWO_DICES
            )
        if score >= 15:
            reward = self.get_config_param(
                KniffelConfig.CHANCE, KniffelConfig.THREE_DICES
            )
        if score >= 20:
            reward = self.get_config_param(
                KniffelConfig.CHANCE, KniffelConfig.FOUR_DICES
            )
        if score >= 25:
            reward = self.get_config_param(
                KniffelConfig.CHANCE, KniffelConfig.FIVE_DICES
            )

        return reward

    def reward_three_times(self, score: int) -> float:
        reward = 0

        if score >= 25:
            reward = self.get_config_param(
                KniffelConfig.THREE_TIMES, KniffelConfig.FIVE_DICES
            )

        elif score >= 20:
            reward = self.get_config_param(
                KniffelConfig.THREE_TIMES, KniffelConfig.FOUR_DICES
            )

        elif score >= 15:
            reward = self.get_config_param(
                KniffelConfig.THREE_TIMES, KniffelConfig.THREE_DICES
            )

        elif score >= 10:
            reward = self.get_config_param(
                KniffelConfig.THREE_TIMES, KniffelConfig.TWO_DICES
            )

        elif score >= 5:
            reward = self.get_config_param(
                KniffelConfig.THREE_TIMES, KniffelConfig.ONE_DICES
            )

        return reward

    def reward_four_times(self, score) -> float:
        reward = 0

        if score >= 25:
            reward = self.get_config_param(
                KniffelConfig.FOUR_TIMES, KniffelConfig.FIVE_DICES
            )

        elif score >= 20:
            reward = self.get_config_param(
                KniffelConfig.FOUR_TIMES, KniffelConfig.FOUR_DICES
            )

        elif score >= 15:
            reward = self.get_config_param(
                KniffelConfig.FOUR_TIMES, KniffelConfig.THREE_DICES
            )

        elif score >= 10:
            reward = self.get_config_param(
                KniffelConfig.FOUR_TIMES, KniffelConfig.TWO_DICES
            )

        elif score >= 5:
            reward = self.get_config_param(
                KniffelConfig.FOUR_TIMES, KniffelConfig.ONE_DICES
            )

        return reward

    def predict_and_apply(self, action: int):

        # init
        done = False
        slashed = False
        reward = 0.0
        finished_turn = False

        # bonus
        bonus = self.kniffel.is_bonus()

        info = {"finished": False, "error": False}

        # Apply action
        enum_action = EnumAction(action)

        try:
            # Finish Actions
            if EnumAction.FINISH_ONES is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.ONES)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    points = selected_option.points / 1
                    reward += self.rewards_single(KniffelConfig.ONES, int(points))

                finished_turn = True
            if EnumAction.FINISH_TWOS is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.TWOS)
                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    dice_count = selected_option.points / 2
                    reward += self.rewards_single(KniffelConfig.TWOS, int(dice_count))

                finished_turn = True
            if EnumAction.FINISH_THREES is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.THREES)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    dice_count = selected_option.points / 3
                    reward += self.rewards_single(KniffelConfig.THREES, int(dice_count))

                finished_turn = True
            if EnumAction.FINISH_FOURS is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.FOURS)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    dice_count = selected_option.points / 4
                    reward += self.rewards_single(KniffelConfig.FOURS, int(dice_count))

                finished_turn = True
            if EnumAction.FINISH_FIVES is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.FIVES)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    dice_count = selected_option.points / 5
                    reward += self.rewards_single(KniffelConfig.FIVES, int(dice_count))

                finished_turn = True
            if EnumAction.FINISH_SIXES is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.SIXES)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    dice_count = selected_option.points / 6
                    reward += self.rewards_single(KniffelConfig.SIXES, int(dice_count))

                finished_turn = True
            if EnumAction.FINISH_THREE_TIMES is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.THREE_TIMES)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    reward += self.reward_three_times(selected_option.points)

                finished_turn = True
            if EnumAction.FINISH_FOUR_TIMES is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.FOUR_TIMES)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    reward += self.reward_four_times(selected_option.points)

                finished_turn = True
            if EnumAction.FINISH_FULL_HOUSE is enum_action:
                self.kniffel.finish_turn(KniffelOptions.FULL_HOUSE)

                if self.reward_mode == "kniffel":
                    reward += 25
                else:
                    reward += self.get_config_param(
                        KniffelConfig.FULL_HOUSE, KniffelConfig.FIVE_DICES
                    )

                finished_turn = True
            if EnumAction.FINISH_SMALL_STREET is enum_action:
                self.kniffel.finish_turn(KniffelOptions.SMALL_STREET)

                if self.reward_mode == "kniffel":
                    reward += 30
                else:
                    reward += self.get_config_param(
                        KniffelConfig.SMALL_STREET, KniffelConfig.FIVE_DICES
                    )

                finished_turn = True
            if EnumAction.FINISH_LARGE_STREET is enum_action:
                self.kniffel.finish_turn(KniffelOptions.LARGE_STREET)

                if self.reward_mode == "kniffel":
                    reward += 40
                else:
                    reward += self.get_config_param(
                        KniffelConfig.LARGE_STREET, KniffelConfig.FIVE_DICES
                    )

                finished_turn = True
            if EnumAction.FINISH_KNIFFEL is enum_action:
                self.kniffel.finish_turn(KniffelOptions.KNIFFEL)

                if self.reward_mode == "kniffel":
                    reward += 50
                else:
                    reward += self.get_config_param(
                        KniffelConfig.KNIFFEL, KniffelConfig.FIVE_DICES
                    )

                finished_turn = True
            if EnumAction.FINISH_CHANCE is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.CHANCE)

                if self.reward_mode == "kniffel":
                    reward += selected_option.points
                else:
                    points = selected_option.points

                    reward += self.reward_chance(points)

                finished_turn = True

            if EnumAction.FINISH_ONES_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.ONES_SLASH)

                if selected_option.id == KniffelOptions.ONES_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.ONES, KniffelConfig.SLASH
                    )
                slashed = True
                finished_turn = True
            if EnumAction.FINISH_TWOS_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.TWOS_SLASH)

                if selected_option.id == KniffelOptions.TWOS_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.TWOS, KniffelConfig.SLASH
                    )

                finished_turn = True
            if EnumAction.FINISH_THREES_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.THREES_SLASH)

                if selected_option.id == KniffelOptions.THREES_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.THREES, KniffelConfig.SLASH
                    )
                slashed = True
                finished_turn = True
            if EnumAction.FINISH_FOURS_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.FOURS_SLASH)

                if selected_option.id == KniffelOptions.FOURS_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.FOURS, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_FIVES_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.FIVES_SLASH)

                if selected_option.id == KniffelOptions.FIVES_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.FIVES, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_SIXES_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.SIXES_SLASH)

                if selected_option.id == KniffelOptions.SIXES_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.SIXES, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_THREE_TIMES_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(
                    KniffelOptions.THREE_TIMES_SLASH
                )

                if selected_option.id == KniffelOptions.THREE_TIMES_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.THREE_TIMES, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_FOUR_TIMES_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(
                    KniffelOptions.FOUR_TIMES_SLASH
                )

                if selected_option.id == KniffelOptions.FOUR_TIMES_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.FOUR_TIMES, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_FULL_HOUSE_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(
                    KniffelOptions.FULL_HOUSE_SLASH
                )

                if selected_option.id == KniffelOptions.FULL_HOUSE_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.FULL_HOUSE, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_SMALL_STREET_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(
                    KniffelOptions.SMALL_STREET_SLASH
                )

                if selected_option.id == KniffelOptions.SMALL_STREET_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.SMALL_STREET, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_LARGE_STREET_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(
                    KniffelOptions.LARGE_STREET_SLASH
                )

                if selected_option.id == KniffelOptions.LARGE_STREET_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.LARGE_STREET, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_KNIFFEL_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.KNIFFEL_SLASH)

                if selected_option.id == KniffelOptions.KNIFFEL_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.KNIFFEL, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True
            if EnumAction.FINISH_CHANCE_SLASH is enum_action:
                selected_option = self.kniffel.finish_turn(KniffelOptions.CHANCE_SLASH)

                if selected_option.id == KniffelOptions.CHANCE_SLASH.value:
                    reward += self.get_config_param(
                        KniffelConfig.CHANCE, KniffelConfig.SLASH
                    )

                slashed = True
                finished_turn = True

            # Continue enum_actions
            if EnumAction.NEXT_0 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_1 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_2 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_3 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_4 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_5 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_6 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_7 is enum_action:
                self.kniffel.add_turn(keep=[0, 0, 1, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_8 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_9 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_10 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_11 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_12 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_13 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_14 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_15 is enum_action:
                self.kniffel.add_turn(keep=[0, 1, 1, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_16 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_17 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_18 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_19 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_20 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_21 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_22 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_23 is enum_action:
                self.kniffel.add_turn(keep=[1, 0, 1, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_24 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_25 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_26 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_27 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 0, 1, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_28 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 0])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_29 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 0, 1])
                reward += self._reward_roll_dice
            if EnumAction.NEXT_30 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 0])
                reward += self._reward_roll_dice

        except Exception as e:
            if e.args[0] == "Game finished!":
                done = True

                if self.kniffel.get_points() > self._reward_finish:
                    reward += self.kniffel.get_points()
                else:
                    reward += self._reward_finish

                # reward += self._reward_finish

                info = {
                    "finished": True,
                    "error": False,
                    "exception": True,
                    "exception_description": str(e),
                }

                if self.logging:
                    print(f"   Game finished: {e}")
                    print("   " + str(info))

            else:
                done = True
                reward += self._reward_game_over

                info = {
                    "finished": True,
                    "error": True,
                    "exception": True,
                    "exception_description": str(e),
                }

                if self.logging:
                    print(f"   Error in game: {e}")
                    print("   " + str(info))

        if self.reward_mode == "kniffel":
            if (
                not slashed
                and not done
                and finished_turn
                and enum_action
                in [
                    enum_action.FINISH_ONES,
                    enum_action.FINISH_TWOS,
                    enum_action.FINISH_THREES,
                    enum_action.FINISH_FOURS,
                    enum_action.FINISH_FIVES,
                    enum_action.FINISH_SIXES,
                ]
                and self.kniffel.is_bonus()
            ):
                # Add bonus to reward
                print("Bonus reached for kniffel reward mode.")
                reward += self._reward_bonus

        elif self.reward_mode == "custom":
            if not bonus and self.kniffel.is_bonus():

                # Add bonus to reward
                print("Bonus reached for custom reward mode.")
                reward += self._reward_bonus

        return reward, done, info

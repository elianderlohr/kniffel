from tokenize import Triple
from gym import Env
import gym.spaces as spaces
import numpy as np
from enum import Enum
import os
import sys
import inspect
import csv
from sympy import elliptic_f
import tensorflow as tf
from kniffel.classes import kniffel
from kniffel.classes.dice_set import DiceSet

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
)

from kniffel.classes.options import KniffelOptions
from kniffel.classes.kniffel import Kniffel
import kniffel.classes.custom_exceptions as ex


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
    NEXT_31 = 44  # 31  	1	1	1	1	1

    # Finish Actions
    FINISH_ONES_SLASH = 45
    FINISH_TWOS_SLASH = 46
    FINISH_THREES_SLASH = 47
    FINISH_FOURS_SLASH = 48
    FINISH_FIVES_SLASH = 49
    FINISH_SIXES_SLASH = 50
    FINISH_THREE_TIMES_SLASH = 51
    FINISH_FOUR_TIMES_SLASH = 52
    FINISH_FULL_HOUSE_SLASH = 53
    FINISH_SMALL_STREET_SLASH = 54
    FINISH_LARGE_STREET_SLASH = 55
    FINISH_KNIFFEL_SLASH = 56
    FINISH_CHANCE_SLASH = 57

    FINISH_GAME = 99


class KniffelEnv(Env):

    kniffel = None
    logging = False

    def __init__(
        self,
        env_config,
        config_file_path="Kniffel.CSV",
        logging=False,
        reward_step=0,
        reward_round=0,
        reward_roll_dice=0.1,
        reward_game_over=-20,
        reward_slash=-5,
        reward_bonus=2,
        reward_finish=10,
    ):
        """Initialize Kniffel Envioronment"""
        self.kniffel = Kniffel(logging=logging)
        # Actions we can take
        self.action_space = spaces.Discrete(57)

        self.observation_space = spaces.Box(
            low=0, high=30, shape=(1, 41), dtype=np.int32
        )

        self.logging = logging

        # Set start
        self.state = self.kniffel.get_state()

        # Info:
        # - https://brefeld.homepage.t-online.de/kniffel.html
        # - https://brefeld.homepage.t-online.de/kniffel-strategie.html
        with open(config_file_path, "r") as file:
            reader = csv.reader(file, delimiter=";")
            d = {v[0]: v[1:] for v in reader}

            header_row = d["Kategorie"]

            del d["Kategorie"]
            del d["Ergebnis"]

            self.config = {
                k: {
                    header_row[i]: float(v1.replace(",", ".").replace("#ZAHL!", "-1"))
                    for i, v1 in enumerate(v)
                }
                for k, v in d.items()
            }

        self._reward_step = self.put_parameter(env_config, "reward_step", reward_step)
        self._reward_round = self.put_parameter(
            env_config, "reward_round", reward_round
        )
        self._reward_roll_dice = self.put_parameter(
            env_config, "reward_roll_dice", reward_roll_dice
        )
        self._reward_game_over = self.put_parameter(
            env_config, "reward_game_over", reward_game_over
        )
        self._reward_slash = self.put_parameter(
            env_config, "reward_slash", reward_slash
        )
        self._reward_bonus = self.put_parameter(
            env_config, "reward_bonus", reward_bonus
        )
        self._reward_finish = self.put_parameter(
            env_config, "reward_finish", reward_finish
        )

    def put_parameter(self, parameters: dict, key: str, alternative: str):
        """Take passed parameter or take alternative

        :param parameters: dict of parameters
        :param key: key of the parameter
        :param alternative: default value for the parameter
        :return: value of the parameter
        """
        if key in parameters.keys():
            return parameters[key]
        else:
            return alternative

    def rewards_single(self, dice_count) -> float:
        """Calculate reward based on amount of dices used for finishing the round.

        :param dice_count: amount of dices
        :return: reward
        """
        if dice_count == 0:
            return float(self.config["Einser"]["0"])
        elif dice_count == 1:
            return float(self.config["Einser"]["1"])
        elif dice_count == 2:
            return float(self.config["Einser"]["2"])
        elif dice_count == 3:
            return float(self.config["Einser"]["3"])
        elif dice_count == 4:
            return float(self.config["Einser"]["4"])
        elif dice_count == 5:
            return float(self.config["Einser"]["perfect"])

    def reward_chance(self, score) -> float:
        reward = 0
        if score >= 5:
            reward = float(self.config["Chance"]["0"])
        if score >= 10:
            reward = float(self.config["Chance"]["1"])
        if score >= 15:
            reward = float(self.config["Chance"]["2"])
        if score >= 20:
            reward = float(self.config["Chance"]["3"])
        if score >= 25:
            reward = float(self.config["Chance"]["4"])
        if score >= 30:
            reward = float(self.config["Chance"]["perfect"])

        return reward

    def mock(self, dices: list):
        print(f"Mock dice: {dices}")
        self.kniffel.mock(DiceSet(dices))

    def step(self, action):
        reward = 0.0
        has_bonus = self.kniffel.is_bonus()

        if self.logging:
            print(self.kniffel.get_state())

        finished_turn = False
        done = False
        # Apply action
        enum_action = EnumAction(action)
        try:
            # Finish Actions
            if EnumAction.FINISH_ONES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.ONES) / 1
                reward += self.rewards_single(points)
                finished_turn = True
            if EnumAction.FINISH_TWOS is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.TWOS) / 2
                reward += self.rewards_single(points)
                finished_turn = True
            if EnumAction.FINISH_THREES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.THREES) / 3
                reward += self.rewards_single(points)
                finished_turn = True
            if EnumAction.FINISH_FOURS is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FOURS) / 4
                reward += self.rewards_single(points)
                finished_turn = True
            if EnumAction.FINISH_FIVES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FIVES) / 5
                reward += self.rewards_single(points)
                finished_turn = True
            if EnumAction.FINISH_SIXES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.SIXES) / 6
                reward += self.rewards_single(points)
                finished_turn = True
            if EnumAction.FINISH_THREE_TIMES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.THREE_TIMES)
                reward += float(self.config["Dreier Pasch"]["3"])
                finished_turn = True
            if EnumAction.FINISH_FOUR_TIMES is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.FOUR_TIMES)
                reward += float(self.config["Vierer Pasch"]["4"])
                finished_turn = True
            if EnumAction.FINISH_FULL_HOUSE is enum_action:
                self.kniffel.finish_turn(KniffelOptions.FULL_HOUSE)
                reward += float(self.config["Full House"]["perfect"])
                finished_turn = True
            if EnumAction.FINISH_SMALL_STREET is enum_action:
                self.kniffel.finish_turn(KniffelOptions.SMALL_STREET)
                reward += float(self.config["Kleine Strasse"]["perfect"])
                finished_turn = True
            if EnumAction.FINISH_LARGE_STREET is enum_action:
                self.kniffel.finish_turn(KniffelOptions.LARGE_STREET)
                reward += float(self.config["Grosse Strasse"]["perfect"])
                finished_turn = True
            if EnumAction.FINISH_KNIFFEL is enum_action:
                self.kniffel.finish_turn(KniffelOptions.KNIFFEL)
                reward += float(self.config["Kniffel"]["perfect"])
                finished_turn = True
            if EnumAction.FINISH_CHANCE is enum_action:
                points = self.kniffel.finish_turn(KniffelOptions.CHANCE)
                reward += self.reward_chance(points)
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
            if EnumAction.NEXT_31 is enum_action:
                self.kniffel.add_turn(keep=[1, 1, 1, 1, 1])
                reward += self._reward_roll_dice

            if EnumAction.FINISH_ONES_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.ONES_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_TWOS_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.TWOS_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_THREES_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.THREES_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_FOURS_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.FOURS_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_FIVES_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.FIVES_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_SIXES_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.SIXES_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_THREE_TIMES_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.THREE_TIMES_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_FOUR_TIMES_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.FOUR_TIMES_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_FULL_HOUSE_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.FULL_HOUSE_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_SMALL_STREET_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.SMALL_STREET_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_LARGE_STREET_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.LARGE_STREET_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_KNIFFEL_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.KNIFFEL_SLASH)
                reward += self._reward_slash
            if EnumAction.FINISH_CHANCE_SLASH is enum_action:
                self.kniffel.finish_turn(KniffelOptions.CHANCE_SLASH)
                reward += self._reward_slash

            if (
                self.kniffel.is_bonus()
                and has_bonus is False
                and (
                    EnumAction.FINISH_ONES is enum_action
                    or EnumAction.FINISH_ONES is enum_action
                    or EnumAction.FINISH_TWOS is enum_action
                    or EnumAction.FINISH_THREES is enum_action
                    or EnumAction.FINISH_FOURS is enum_action
                    or EnumAction.FINISH_FIVES is enum_action
                    or EnumAction.FINISH_SIXES is enum_action
                )
            ):
                reward += self._reward_bonus
        except BaseException as e:
            if e == ex.GameFinishedException:
                done = True
            else:
                reward += self._reward_game_over
                done = True

        if self.kniffel.turns_left() == 1 and finished_turn:
            reward += self.kniffel.get_points()
            done = True

        self.state = self.kniffel.get_state()

        # Set placeholder for info
        info = {}

        # kniffel_rounds = self.kniffel.get_played_rounds() / 39
        # kniffel_points = self.kniffel.get_points() / 300

        # reward += self._reward_step + kniffel_points + kniffel_rounds

        # Return step information
        return self.state, reward, done, info

    def render(self):
        # Implement viz
        pass

    def reset(self):
        # Reset
        del self.state
        del self.kniffel

        self.kniffel = Kniffel()
        self.state = self.kniffel.get_state()

        return self.state

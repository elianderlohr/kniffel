from enum import Enum


class KniffelOptions(Enum):
    ONES: int = 1
    TWOS: int = 2
    THREES: int = 3
    FOURS: int = 4
    FIVES: int = 5
    SIXES: int = 6
    
    THREE_TIMES: int = 7
    FOUR_TIMES: int = 8
    FULL_HOUSE: int = 9
    SMALL_STREET: int = 10
    LARGE_STREET: int = 11
    KNIFFEL: int = 12
    CHANCE: int = 13

    ONES_SLASH: int = 14
    TWOS_SLASH: int = 15
    THREES_SLASH: int = 16
    FOURS_SLASH: int = 17
    FIVES_SLASH: int = 18
    SIXES_SLASH: int = 19
    THREE_TIMES_SLASH: int = 20
    FOUR_TIMES_SLASH: int = 21
    FULL_HOUSE_SLASH: int = 22
    SMALL_STREET_SLASH: int = 23
    LARGE_STREET_SLASH: int = 24
    KNIFFEL_SLASH: int = 25
    CHANCE_SLASH: int = 26

    DEFAULT: int = 27

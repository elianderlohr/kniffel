from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from env.open_ai_env import KniffelEnv
from src.env.env_helper import KniffelConfig

from src.env.env_helper import EnumAction

env_config = {
    "reward_roll_dice": 0,
    "reward_game_over": -40,
    "reward_finish": 15,
    "reward_bonus": 5,
}


def send_step(dice, env, action, score):
    if len(dice) == 5:
        env.mock(dice)

    state, reward, done, info = env.step(action)

    score += reward

    return score


def test_get_config():
    env = KniffelEnv(env_config, logging=True, config_file_path="src/config/config.csv")

    print(env.kniffel_helper.config)

    assert (
        env.kniffel_helper.get_config_param(KniffelConfig.ONES, KniffelConfig.COLUMN_5)
        == 40
    )


def test_env():
    score = 0

    env = KniffelEnv(env_config, logging=True, config_file_path="src/config/config.csv")

    # try 1
    score = send_step([], env, 13, score)
    score = send_step([1, 1, 1, 1, 1], env, 0, score)

    # try 2
    score = send_step([], env, 13, score)
    score = send_step([2, 2, 2, 2, 2], env, 1, score)

    # try 3
    score = send_step([3, 3, 3, 3, 3], env, 2, score)

    # try 4
    score = send_step([4, 4, 4, 4, 4], env, 3, score)

    # try 5
    score = send_step([5, 5, 5, 5, 5], env, 4, score)

    # try 6
    score = send_step([6, 6, 6, 6, 6], env, 5, score)

    # try 7
    score = send_step([6, 6, 6, 6, 6], env, 6, score)

    # try 8
    score = send_step([6, 6, 6, 6, 6], env, 7, score)

    # try 9
    score = send_step([6, 6, 6, 5, 5], env, 8, score)

    # try 10
    score = send_step([1, 2, 3, 4, 5], env, 9, score)

    # try 11
    score = send_step([1, 2, 3, 4, 5], env, 10, score)

    # try 12
    score = send_step([6, 6, 6, 6, 6], env, 11, score)

    # try 13
    score = send_step([6, 6, 6, 6, 6], env, 12, score)

    assert score == 1361.0


def test_perfect_game():
    score = 0

    env = KniffelEnv(
        env_config, logging=False, config_file_path="src/config/config.csv"
    )

    # try 1
    score = send_step([1, 1, 1, 1, 1], env, 0, score)
    assert score == 41

    # try 2
    score = send_step([2, 2, 2, 2, 2], env, 1, score)
    assert score == 84

    # try 3
    score = send_step([3, 3, 3, 3, 3], env, 2, score)
    assert score == 130

    # try 4
    score = send_step([4, 4, 4, 4, 4], env, 3, score)
    assert score == 160

    # try 5
    score = send_step([5, 5, 5, 5, 5], env, 4, score)
    assert score == 200

    # try 6
    score = send_step([6, 6, 6, 6, 6], env, 5, score)
    assert score == 240

    # try 7
    score = send_step([6, 6, 6, 6, 6], env, 6, score)
    assert score == 280

    # try 8
    score = send_step([6, 6, 6, 6, 6], env, 7, score)
    assert score == 320

    # try 9
    score = send_step([6, 6, 6, 5, 5], env, 8, score)
    assert score == 353

    # try 10
    score = send_step([1, 2, 3, 4, 5], env, 9, score)
    assert score == 393

    # try 11
    score = send_step([1, 2, 3, 4, 5], env, 10, score)
    assert score == 446

    # try 12
    score = send_step([6, 6, 6, 6, 6], env, 11, score)
    assert score == 512

    # try 13
    score = send_step([6, 6, 6, 6, 6], env, 12, score)

    assert score == 512 + 150 + 50 + 30


def test_normal_game():
    score = 0

    env = KniffelEnv(
        env_config, logging=False, config_file_path="src/config/config.csv"
    )

    # try 1
    score = send_step([1, 2, 2, 2, 2], env, 0, score)
    assert score == 1.6
    assert env.kniffel_helper.kniffel.get_points() == 1

    # try 2
    score = send_step([2, 2, 1, 1, 1], env, 1, score)
    assert score == 8
    assert env.kniffel_helper.kniffel.get_points() == 5

    # try 3
    score = send_step([3, 3, 3, 4, 4], env, 2, score)
    assert score == 22
    assert env.kniffel_helper.kniffel.get_points() == 14

    # try 4
    score = send_step([4, 4, 5, 5, 5], env, 3, score)
    assert score == 28.4
    assert env.kniffel_helper.kniffel.get_points() == 22

    # try 5
    score = send_step([5, 5, 5, 5, 5], env, 4, score)
    assert score == 68.4
    assert env.kniffel_helper.kniffel.get_points() == 47

    # try 6
    score = send_step([6, 6, 6, 5, 5], env, 5, score)
    assert score == 82.4
    assert env.kniffel_helper.kniffel.get_points() == 65

    # try 7
    score = send_step([6, 5, 4, 3, 1], env, 51, score)
    assert score == -217.6
    assert env.kniffel_helper.kniffel.get_points() == 65

    # try 8
    score = send_step([6, 6, 6, 6, 6], env, 52, score)
    assert score == -517.6
    assert env.kniffel_helper.kniffel.get_points() == 65

    # try 9
    score = send_step([6, 6, 6, 5, 5], env, 8, score)
    assert score == -484.6
    assert env.kniffel_helper.kniffel.get_points() == 90

    # try 10
    score = send_step([1, 2, 3, 4, 5], env, 9, score)
    assert score == -444.6
    assert env.kniffel_helper.kniffel.get_points() == 120

    # try 11
    score = send_step([1, 2, 3, 4, 5], env, 10, score)
    assert score == -391.6
    assert env.kniffel_helper.kniffel.get_points() == 160

    # try 12
    score = send_step([6, 6, 6, 6, 6], env, 11, score)
    assert score == -325.6
    assert env.kniffel_helper.kniffel.get_points() == 210

    # try 13
    score = send_step([6, 6, 6, 2, 2], env, 12, score)
    assert score == -103.60000000000002
    assert env.kniffel_helper.kniffel.get_points() == 267

    print(env.kniffel_helper.kniffel.get_state())


def test_slash_game():
    score = 0

    env = KniffelEnv(
        env_config, logging=False, config_file_path="src/config/config.csv"
    )

    # try 1
    score = send_step([2, 2, 2, 2, 2], env, 45, score)
    assert score == -50.0

    score = send_step([2, 2, 2, 2, 2], env, 46, score)
    assert score == -150.0

    score = send_step([2, 2, 2, 2, 2], env, 47, score)
    assert score == -300.0

    score = send_step([4, 4, 4, 4, 4], env, 48, score)
    assert score == -500.0

    score = send_step([4, 4, 4, 4, 4], env, 49, score)
    assert score == -750.0

    score = send_step([4, 4, 4, 4, 4], env, 50, score)
    assert score == -1050.0

    score = send_step([4, 4, 4, 4, 4], env, 51, score)
    assert score == -1350.0

    score = send_step([4, 4, 4, 4, 4], env, 52, score)
    assert score == -1650.0

    score = send_step([4, 4, 4, 4, 4], env, 53, score)
    assert score == -1900.0

    score = send_step([4, 4, 4, 4, 4], env, 54, score)
    assert score == -2200.0

    score = send_step([4, 4, 4, 4, 4], env, 55, score)
    assert score == -2600.0

    score = send_step([4, 4, 4, 4, 4], env, 56, score)
    assert score == -3100.0

    score = send_step([6, 6, 6, 6, 6], env, 12, score)
    assert score == -2920.0


def test_broken_game():
    score = 0

    env = KniffelEnv(
        env_config, logging=False, config_file_path="src/config/config.csv"
    )

    # try 1
    score = send_step([], env, 13, score)
    score = send_step([1, 1, 1, 1, 1], env, 2, score)

    assert score == -500


def test_finish_game():
    score = 0

    env = KniffelEnv(
        env_config,
        logging=False,
        config_file_path="src/config/config.csv",
        reward_simple=False,
    )

    # try 1
    score = send_step([1, 1, 1, 1, 1], env, 0, score)
    # assert score == 40

    # try 2
    score = send_step([2, 2, 2, 2, 2], env, 1, score)
    # assert score == 80

    # try 3
    score = send_step([3, 3, 3, 3, 3], env, 2, score)
    # assert score == 120

    # try 4
    score = send_step([4, 4, 4, 4, 4], env, 3, score)
    # assert score == 160

    # try 5
    score = send_step([5, 5, 5, 5, 5], env, 4, score)
    # assert score == 200

    # try 6
    score = send_step([6, 6, 6, 6, 6], env, 5, score)
    # assert score == 240

    # try 7
    score = send_step([6, 6, 6, 6, 6], env, 6, score)
    # assert score == 280

    # try 8
    score = send_step([6, 6, 6, 6, 6], env, 7, score)
    # assert score == 320

    # try 9
    score = send_step([6, 6, 6, 5, 5], env, 8, score)
    # assert score == 353

    # try 10
    score = send_step([1, 2, 3, 4, 5], env, 9, score)
    # assert score == 393

    # try 11
    score = send_step([1, 2, 3, 4, 5], env, 10, score)
    # assert score == 446

    # try 12
    score = send_step([6, 6, 6, 6, 6], env, 11, score)
    # assert score == 512

    # try 13
    score = send_step([6, 6, 6, 6, 6], env, 12, score)
    assert score == 742

    # Reset Env
    env.reset()

    score = 0
    score = send_step([1, 1, 1, 1, 1], env, 0, score)

    # assert score == 40
    assert env.kniffel_helper.kniffel.get_length() == 2


def test_slash_game():
    score = 0

    env = KniffelEnv(
        env_config,
        logging=False,
        config_file_path="src/config/config.csv",
        reward_simple=False,
    )

    # try 1
    score = send_step([1, 1, 1, 1, 1], env, 44, score)
    # assert score == 40

    # try 2
    score = send_step([2, 2, 2, 2, 2], env, 45, score)
    # assert score == 80

    # try 3
    score = send_step([3, 3, 3, 3, 3], env, 46, score)
    # assert score == 120

    # try 4
    score = send_step([4, 4, 4, 4, 4], env, 47, score)
    # assert score == 160

    # try 5
    score = send_step([5, 5, 5, 5, 5], env, 48, score)
    # assert score == 200

    # try 6
    score = send_step([6, 6, 6, 6, 6], env, 49, score)
    # assert score == 240

    # try 7
    score = send_step([6, 6, 6, 6, 6], env, 50, score)
    # assert score == 280

    # try 8
    score = send_step([6, 6, 6, 6, 6], env, 51, score)
    # assert score == 320

    # try 9
    score = send_step([6, 6, 6, 5, 5], env, 52, score)
    # assert score == 353

    # try 10
    score = send_step([1, 2, 3, 4, 5], env, 53, score)
    # assert score == 393

    # try 11
    score = send_step([1, 2, 3, 4, 5], env, 54, score)
    # assert score == 446

    # try 12
    score = send_step([6, 6, 6, 6, 6], env, 55, score)
    # assert score == 512

    # try 13
    score = send_step([6, 6, 6, 6, 6], env, 56, score)
    assert score == 742


def apply_and_reset(env, action, dices, reset=True):
    score = send_step(dices, env, action, 0)

    if reset:
        env.reset()

    return score




def test_individual():
    env = KniffelEnv(
        env_config,
        logging=False,
        config_file_path="src/config/config.csv",
        reward_simple=False,
    )

    # try ones
    assert apply_and_reset(env, EnumAction.FINISH_ONES, [1, 1, 1, 1, 1]) == 10
    assert apply_and_reset(env, EnumAction.FINISH_ONES, [1, 1, 1, 1, 6]) == 4
    assert apply_and_reset(env, EnumAction.FINISH_ONES, [1, 1, 1, 6, 6]) == 0.5
    assert apply_and_reset(env, EnumAction.FINISH_ONES, [1, 1, 6, 6, 6]) == 0.2
    assert apply_and_reset(env, EnumAction.FINISH_ONES, [1, 6, 6, 6, 6]) == 0.1

    # try twos
    assert apply_and_reset(env, EnumAction.FINISH_TWOS, [2, 2, 2, 2, 2]) == 13
    assert apply_and_reset(env, EnumAction.FINISH_TWOS, [2, 2, 2, 2, 6]) == 5
    assert apply_and_reset(env, EnumAction.FINISH_TWOS, [2, 2, 2, 6, 6]) == 0.6
    assert apply_and_reset(env, EnumAction.FINISH_TWOS, [2, 2, 6, 6, 6]) == 0.3
    assert apply_and_reset(env, EnumAction.FINISH_TWOS, [2, 6, 6, 6, 6]) == 0.2

    # try threes
    assert apply_and_reset(env, EnumAction.FINISH_THREES, [3, 3, 3, 3, 3]) == 16
    assert apply_and_reset(env, EnumAction.FINISH_THREES, [3, 3, 3, 3, 6]) == 6.5
    assert apply_and_reset(env, EnumAction.FINISH_THREES, [3, 3, 3, 6, 6]) == 0.7
    assert apply_and_reset(env, EnumAction.FINISH_THREES, [3, 3, 6, 6, 6]) == 0.4
    assert apply_and_reset(env, EnumAction.FINISH_THREES, [3, 6, 6, 6, 6]) == 0.3

    # try fours
    assert apply_and_reset(env, EnumAction.FINISH_FOURS, [4, 4, 4, 4, 4]) == 19
    assert apply_and_reset(env, EnumAction.FINISH_FOURS, [4, 4, 4, 4, 6]) == 8
    assert apply_and_reset(env, EnumAction.FINISH_FOURS, [4, 4, 4, 6, 6]) == 0.8
    assert apply_and_reset(env, EnumAction.FINISH_FOURS, [4, 4, 6, 6, 6]) == 0.5
    assert apply_and_reset(env, EnumAction.FINISH_FOURS, [4, 6, 6, 6, 6]) == 0.4

    # try fives
    assert apply_and_reset(env, EnumAction.FINISH_FIVES, [5, 5, 5, 5, 5]) == 22
    assert apply_and_reset(env, EnumAction.FINISH_FIVES, [5, 5, 5, 5, 6]) == 8.5
    assert apply_and_reset(env, EnumAction.FINISH_FIVES, [5, 5, 5, 6, 6]) == 0.9
    assert apply_and_reset(env, EnumAction.FINISH_FIVES, [5, 5, 6, 6, 6]) == 0.8
    assert apply_and_reset(env, EnumAction.FINISH_FIVES, [5, 6, 6, 6, 6]) == 0.5

    # try sixes
    assert apply_and_reset(env, EnumAction.FINISH_SIXES, [6, 6, 6, 6, 6]) == 25
    assert apply_and_reset(env, EnumAction.FINISH_SIXES, [6, 6, 6, 6, 5]) == 9
    assert apply_and_reset(env, EnumAction.FINISH_SIXES, [6, 6, 6, 5, 5]) == 1
    assert apply_and_reset(env, EnumAction.FINISH_SIXES, [6, 6, 5, 5, 5]) == 0.7
    assert apply_and_reset(env, EnumAction.FINISH_SIXES, [6, 5, 5, 5, 5]) == 0.6

    # try three of a kind
    assert apply_and_reset(env, EnumAction.FINISH_THREE_TIMES, [1, 1, 1, 1, 1]) == 1
    assert apply_and_reset(env, EnumAction.FINISH_THREE_TIMES, [2, 2, 2, 1, 1]) == 1
    assert apply_and_reset(env, EnumAction.FINISH_THREE_TIMES, [3, 3, 3, 1, 1]) == 3
    assert apply_and_reset(env, EnumAction.FINISH_THREE_TIMES, [4, 4, 4, 1, 1]) == 3
    assert apply_and_reset(env, EnumAction.FINISH_THREE_TIMES, [5, 5, 5, 1, 1]) == 5
    assert apply_and_reset(env, EnumAction.FINISH_THREE_TIMES, [6, 6, 6, 1, 1]) == 10
    assert apply_and_reset(env, EnumAction.FINISH_THREE_TIMES, [6, 6, 6, 5, 5]) == 23

    # try four of a kind
    assert apply_and_reset(env, EnumAction.FINISH_FOUR_TIMES, [1, 1, 1, 1, 1]) == 1
    assert apply_and_reset(env, EnumAction.FINISH_FOUR_TIMES, [6, 6, 6, 6, 1]) == 25
    assert apply_and_reset(env, EnumAction.FINISH_FOUR_TIMES, [6, 6, 6, 6, 6]) == 25

    # try negative four of a kind
    assert apply_and_reset(env, EnumAction.FINISH_FOUR_TIMES, [1, 1, 1, 2, 2]) == -15

    # try chance
    assert apply_and_reset(env, EnumAction.FINISH_CHANCE, [1, 1, 1, 1, 1]) == 0.2
    assert apply_and_reset(env, EnumAction.FINISH_CHANCE, [2, 2, 2, 2, 2]) == 1
    assert apply_and_reset(env, EnumAction.FINISH_CHANCE, [3, 3, 3, 3, 3]) == 3
    assert apply_and_reset(env, EnumAction.FINISH_CHANCE, [4, 4, 4, 4, 4]) == 8
    assert apply_and_reset(env, EnumAction.FINISH_CHANCE, [5, 5, 5, 5, 5]) == 20
    assert apply_and_reset(env, EnumAction.FINISH_CHANCE, [6, 6, 6, 6, 6]) == 20

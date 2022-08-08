from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.ai.env import KniffelEnv


env_config = {
    "reward_roll_dice": 0,
    "reward_game_over": -500,
    "reward_finish": 150,
    "reward_bonus": 50,
}


def send_step(dice, env, action, score):
    if len(dice) == 5:
        env.mock(dice)

    n_state, reward, done, info = env.step(action)

    score += reward
    # print(f"Score: {score}")

    return score


def test_env():
    score = 0

    env = KniffelEnv(
        env_config, logging=False, config_file_path="src/config/Kniffel.CSV"
    )

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

    assert score == 742


def test_perfect_game():
    score = 0

    env = KniffelEnv(
        env_config, logging=True, config_file_path="src/config/Kniffel.CSV"
    )

    # try 1
    score = send_step([1, 1, 1, 1, 1], env, 0, score)
    assert score == 40

    # try 2
    score = send_step([2, 2, 2, 2, 2], env, 1, score)
    assert score == 80

    # try 3
    score = send_step([3, 3, 3, 3, 3], env, 2, score)
    assert score == 120

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


def test_slash_game():
    score = 0

    env = KniffelEnv(
        env_config, logging=False, config_file_path="src/config/Kniffel.CSV"
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
        env_config, logging=False, config_file_path="src/config/Kniffel.CSV"
    )

    # try 1
    score = send_step([], env, 13, score)
    score = send_step([1, 1, 1, 1, 1], env, 2, score)

    assert score == -500

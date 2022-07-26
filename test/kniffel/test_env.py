import pytest

from pathlib import Path
import sys

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.ai.env import KniffelEnv


def send_step(dice, env, action, score):
    if len(dice) > 0:
        env.mock(dice)
    n_state, reward, done, info = env.step(action)

    score += reward
    # print(f"Score: {score}")

    return score


def test_env():
    env_config = {
        "reward_step": 0,
        "reward_roll_dice": 0.5,
        "reward_game_over": -1000,
        "reward_slash": -10,
        "reward_bonus": 20,
        "reward_finish": 50,
    }

    score = 0

    env = KniffelEnv(env_config, logging=False, config_file_path="src/ai/Kniffel.CSV")

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

    assert score == 18871.144615384615

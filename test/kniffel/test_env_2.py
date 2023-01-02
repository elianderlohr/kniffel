from src.env.open_ai_env import KniffelEnv
from src.env.env_helper import EnumAction


def send_step(dice, env, action, score):
    if len(dice) == 5:
        env.mock(dice)

    state, reward, done, info = env.step(action)

    score += reward

    return score


def test_bonus():
    env_config = {
        "reward_roll_dice": 0,
        "reward_game_over": -25,
        "reward_finish": 25,
        "reward_bonus": 25,
        "reward_mode": "custom",
        "state_mode": "continuous",
    }

    env = KniffelEnv(
        env_config,
        logging=False,
        config_file_path="src/config/config.csv",
        reward_mode="custom",
        state_mode="continuous",
    )

    # Ones
    score = send_step([1, 1, 1, 1, 1], env, EnumAction.FINISH_ONES, 0)
    assert score == 10

    # Twos
    score = send_step([2, 2, 2, 2, 2], env, EnumAction.FINISH_TWOS, score)
    assert score == 13 + (10)  # = 23

    # Threes
    score = send_step([3, 3, 3, 3, 3], env, EnumAction.FINISH_THREES, score)
    assert score == 16 + (10 + 13)  # = 39

    # Fours
    score = send_step([4, 4, 4, 4, 4], env, EnumAction.FINISH_FOURS, score)
    assert score == 19 + (10 + 13 + 16)  # = 58

    # Fives
    score = send_step([5, 5, 5, 5, 5], env, EnumAction.FINISH_FIVES, score)
    assert score == 22 + env_config["reward_bonus"] + (
        10 + 13 + 16 + 19
    )  # = 80 + 25 (bonus)

    # Sixes
    score = send_step([6, 6, 6, 6, 6], env, EnumAction.FINISH_SIXES, score)
    assert score == 25 + env_config["reward_bonus"] + (
        10 + 13 + 16 + 19 + 22 + env_config["reward_bonus"]
    )  # = 130 + 25 (bonus)

    # Three of a kind
    score = send_step([6, 6, 6, 6, 6], env, EnumAction.FINISH_THREE_TIMES, score)
    assert score == 23 + (155)  # = 183

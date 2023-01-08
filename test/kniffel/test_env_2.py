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
        "reward_bonus": 250,
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
    assert score == 25  # 5

    # Twos
    score = send_step([2, 2, 2, 2, 2], env, EnumAction.FINISH_TWOS, score)
    assert score == 25 + (25)  # 10 + 5 = 15

    # Threes
    score = send_step([3, 3, 3, 3, 3], env, EnumAction.FINISH_THREES, score)
    assert score == 25 + (25 + 25)  # = 5 * 3 + 15 = 30

    # Fours
    score = send_step([4, 4, 4, 4, 4], env, EnumAction.FINISH_FOURS, score)
    assert score == 25 + (25 + 25 + 25)  # = 5 * 4 + 30 = 45

    # Fives
    score = send_step([5, 5, 5, 5, 5], env, EnumAction.FINISH_FIVES, score)
    assert score == 25 + env_config["reward_bonus"] + (
        25 + 25 + 25 + 25
    )  # = 5 * 5 + 45 = 70 > bonus should apply

    # Sixes
    score = send_step([6, 6, 6, 6, 6], env, EnumAction.FINISH_SIXES, score)
    assert score == 25 + (
        25 + 25 + 25 + 25 + 25 + env_config["reward_bonus"]
    )  # = 130 + 25 (bonus)

    # Three of a kind
    score = send_step([6, 6, 6, 6, 6], env, EnumAction.FINISH_THREE_TIMES, score)
    assert score == 23 + (
        25 + 25 + 25 + 25 + 25 + 25 + env_config["reward_bonus"]
    )  # = 183

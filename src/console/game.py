import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from pathlib import Path
import sys


path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.kniffel.classes.kniffel import Kniffel
from src.ai.ai import KniffelAI
from src.ai.draw import KniffelDraw
from src.kniffel.classes.dice_set import DiceSet
from src.ai.env import EnumAction
from src.kniffel.classes.status import KniffelStatus

if __name__ == "__main__":
    print(
        """ _  ___   _ _____ ______ ______ ______ _      
| |/ / \ | |_   _|  ____|  ____|  ____| |     
| ' /|  \| | | | | |__  | |__  | |__  | |     
|  < | . ` | | | |  __| |  __| |  __| | |     
| . \| |\  |_| |_| |    | |    | |____| |____ 
|_|\_\_| \_|_____|_|    |_|    |______|______|"""
    )

    print("")

    if input("Do you want to play a new game? (y/n)") == "y":

        ai = KniffelAI(
            load=False,
            config_path="src/config/Kniffel.CSV",
            path_prefix="",
            hyperparater_base=None,
            env_observation_space=24,
            env_action_space=58,
        )

        env_config = {
            "reward_roll_dice": 0,
            "reward_game_over": -1000,
            "reward_finish": 150,
            "reward_bonus": 50,
        }

        agent = ai.build_use_agent(
            path="output/weights/model_4",
            episodes=1,
            env_config=env_config,
            weights_name="weights",
            logging=False,
        )

        if input("Do you want to use your own dices? (y/n)") == "y":
            kniffel = Kniffel(custom=True)

            while True:

                if (
                    kniffel.is_new_game()
                    or kniffel.get_last().status.value == KniffelStatus.INIT.value
                ):
                    dices = input(
                        "Input your dices in the following style: '1 5 3 2 4' > "
                    )
                    dices_list = [int(d) for d in dices.split(" ")]

                    kniffel.mock(DiceSet(mock=dices_list))
                elif kniffel.get_last().status.value == KniffelStatus.ATTEMPTING.value:
                    state = kniffel.get_state()
                    print()
                    print(KniffelDraw().draw_dices(state[0][0:5]))
                    print()
                    print(KniffelDraw().draw_sheet(kniffel, state[0][5:]))

                action = agent.forward(kniffel.get_state())
                print(
                    f"The AI suggest that you do the following action: {EnumAction(action)}"
                )

                if input("Do you want to accept this action? (y/n)") == "y":
                    ai.apply_prediction(kniffel, EnumAction(action))

        else:
            kniffel = Kniffel()
            while True:
                state = kniffel.get_state()
                print()
                print(KniffelDraw().draw_dices(state[0][0:5]))
                print()
                print(KniffelDraw().draw_sheet(kniffel, state[0][5:]))

                action = agent.forward(state)
                enum_action = EnumAction(action)

                print(f"The AI suggest that you do the following action: {enum_action}")

                if input("Do you want to accept this action? (y/n)") == "y":
                    ai.apply_prediction(kniffel, EnumAction(action))
                else:
                    new_action = input("Give the id of the Action: ")
                    new_enum_action = EnumAction(int(new_action))

                    ai.apply_prediction(kniffel, EnumAction(new_enum_action))

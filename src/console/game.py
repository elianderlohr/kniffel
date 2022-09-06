import warnings

warnings.filterwarnings("ignore")

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
from pathlib import Path

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from src.rl.rl import KniffelRL
from src.kniffel.classes.dice_set import DiceSet
from src.kniffel.classes.kniffel import Kniffel
from src.kniffel.classes.status import KniffelStatus
from src.rl.env import EnumAction
from utils.draw import KniffelDraw


def print_kniffel():
    print(
        """
     __  ___ .__   __.  __   _______  _______  _______  __                    .______       __      
    |  |/  / |  \ |  | |  | |   ____||   ____||   ____||  |                   |   _  \     |  |     
    |  '  /  |   \|  | |  | |  |__   |  |__   |  |__   |  |         ______    |  |_)  |    |  |     
    |    <   |  . `  | |  | |   __|  |   __|  |   __|  |  |        |______|   |      /     |  |     
    |  .  \  |  |\   | |  | |  |     |  |     |  |____ |  `----.              |  |\  \----.|  `----.
    |__|\__\ |__| \__| |__| |__|     |__|     |_______||_______|              | _| `._____||_______|
                                                                                                
"""
    )


def print_round(round: int):
    if round == 1:
        print(
            """
     __                              
    |__)   _         _    _|  .   /| 
    | \   (_)  |_|  | )  (_|  .    | 
                                 
"""
        )
    elif round == 2:
        print(
            """
     __                           __  
    |__)   _         _    _|  .    _) 
    | \   (_)  |_|  | )  (_|  .   /__ 
                                                                   
"""
        )
    elif round == 3:
        print(
            """
     __                           __  
    |__)   _         _    _|  .    _) 
    | \   (_)  |_|  | )  (_|  .   __) 
                                  
                                 
"""
        )
    elif round == 4:
        print(
            """
     __                                
    |__)   _         _    _|  .   |__| 
    | \   (_)  |_|  | )  (_|  .      | 
                                   
"""
        )
    elif round == 5:
        print(
            """
     __                            __ 
    |__)   _         _    _|  .   |_  
    | \   (_)  |_|  | )  (_|  .   __) 
                                  
"""
        )
    elif round == 6:
        print(
            """
     __                            __  
    |__)   _         _    _|  .   /__  
    | \   (_)  |_|  | )  (_|  .   \__) 
                                   
"""
        )
    elif round == 7:
        print(
            """
     __                           ___ 
    |__)   _         _    _|  .     / 
    | \   (_)  |_|  | )  (_|  .    /  
                                  
"""
        )
    elif round == 8:
        print(
            """
     __                            __  
    |__)   _         _    _|  .   (__) 
    | \   (_)  |_|  | )  (_|  .   (__) 
                                   
"""
        )
    elif round == 9:
        print(
            """
     __                            __  
    |__)   _         _    _|  .   (__\ 
    | \   (_)  |_|  | )  (_|  .    __/ 
                                   
"""
        )
    elif round == 10:
        print(
            """
     __                               __  
    |__)   _         _    _|  .   /| /  \ 
    | \   (_)  |_|  | )  (_|  .    | \__/ 
                                      
"""
        )
    elif round == 11:
        print(
            """
     __                                 
    |__)   _         _    _|  .   /| /| 
    | \   (_)  |_|  | )  (_|  .    |  | 
                                    
"""
        )
    elif round == 12:
        print(
            """
     __                              __  
    |__)   _         _    _|  .   /|  _) 
    | \   (_)  |_|  | )  (_|  .    | /__ 
                                     
"""
        )
    elif round == 13:
        print(
            """
     __                              __  
    |__)   _         _    _|  .   /|  _) 
    | \   (_)  |_|  | )  (_|  .    | __) 
                                     
"""
        )
    else:
        print("Round: " + str(round))


def print_header(kniffel: Kniffel, new: bool = False, show_dice: bool = False):
    clear = lambda: os.system("cls")
    clear()
    print_kniffel()
    print()
    print(
        "_____________________________________________________________________________"
    )
    if kniffel.is_new_game() is False:
        print_round(13 - kniffel.turns_left() + 1)
    else:
        print_round(1)
    print()
    print("    Throws left: 3 throws")
    print()
    if show_dice:
        print(KniffelDraw().draw_dices(kniffel.get_state()[0][0:5]))
    if new is False:
        state = kniffel.get_state()
        if show_dice is False:
            print(KniffelDraw().draw_dices(kniffel.get_state()[0][0:5]))
        print()
        print(KniffelDraw().draw_sheet(kniffel))


def get_action(action: EnumAction) -> str:
    if action.value is EnumAction.FINISH_ONES.value:
        return "Finish the ones."
    if action.value is EnumAction.FINISH_TWOS.value:
        return "Finish the two."
    if action.value is EnumAction.FINISH_THREES.value:
        return "Finish the threes."
    if action.value is EnumAction.FINISH_FOURS.value:
        return "Finish the fours."
    if action.value is EnumAction.FINISH_FIVES.value:
        return "Finish the fives."
    if action.value is EnumAction.FINISH_SIXES.value:
        return "Finish the sixes."
    if action.value is EnumAction.FINISH_THREE_TIMES.value:
        return "Finish the three times."
    if action.value is EnumAction.FINISH_FOUR_TIMES.value:
        return "Finish the four times."
    if action.value is EnumAction.FINISH_FULL_HOUSE.value:
        return "Finish the full house."
    if action.value is EnumAction.FINISH_SMALL_STREET.value:
        return "Finish the small street."
    if action.value is EnumAction.FINISH_LARGE_STREET.value:
        return "Finish the large street."
    if action.value is EnumAction.FINISH_KNIFFEL.value:
        return "Finish the kniffel."
    if action.value is EnumAction.FINISH_CHANCE.value:
        return "Finish the chance."

    # Continue Actions
    if action.value is EnumAction.NEXT_0.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 0, 0"
    if action.value is EnumAction.NEXT_1.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 0, 1"
    if action.value is EnumAction.NEXT_2.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 1, 0"
    if action.value is EnumAction.NEXT_3.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 1, 1"
    if action.value is EnumAction.NEXT_4.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 0, 0"
    if action.value is EnumAction.NEXT_5.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 0, 1"
    if action.value is EnumAction.NEXT_6.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 1, 0"
    if action.value is EnumAction.NEXT_7.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 1, 1"
    if action.value is EnumAction.NEXT_8.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 0, 0"
    if action.value is EnumAction.NEXT_9.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 0, 1"
    if action.value is EnumAction.NEXT_10.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 1, 0"
    if action.value is EnumAction.NEXT_11.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 1, 1"
    if action.value is EnumAction.NEXT_12.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 0, 0"
    if action.value is EnumAction.NEXT_13.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 0, 1"
    if action.value is EnumAction.NEXT_14.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 1, 0"
    if action.value is EnumAction.NEXT_15.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 1, 1"
    if action.value is EnumAction.NEXT_16.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 0, 0"
    if action.value is EnumAction.NEXT_17.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 0, 1"
    if action.value is EnumAction.NEXT_18.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 1, 0"
    if action.value is EnumAction.NEXT_19.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 1, 1"
    if action.value is EnumAction.NEXT_20.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 0, 0"
    if action.value is EnumAction.NEXT_21.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 0, 1"
    if action.value is EnumAction.NEXT_22.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 1, 0"
    if action.value is EnumAction.NEXT_23.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 1, 1"
    if action.value is EnumAction.NEXT_24.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 0, 0"
    if action.value is EnumAction.NEXT_25.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 0, 1"
    if action.value is EnumAction.NEXT_26.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 1, 0"
    if action.value is EnumAction.NEXT_27.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 1, 1"
    if action.value is EnumAction.NEXT_28.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 1, 0, 0"
    if action.value is EnumAction.NEXT_29.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 1, 0, 1"
    if action.value is EnumAction.NEXT_30.value:
        return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 1, 1, 0"
    # if action.value is EnumAction.NEXT_31.value:
    #    return "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 1, 1, 1"

    # if action.value is EnumAction.FINISH Actions
    if action.value is EnumAction.FINISH_ONES_SLASH.value:
        return "Slash the ones."
    if action.value is EnumAction.FINISH_TWOS_SLASH.value:
        return "Slash the two."
    if action.value is EnumAction.FINISH_THREES_SLASH.value:
        return "Slash the threes."
    if action.value is EnumAction.FINISH_FOURS_SLASH.value:
        return "Slash the fours."
    if action.value is EnumAction.FINISH_FIVES_SLASH.value:
        return "Slash the fives."
    if action.value is EnumAction.FINISH_SIXES_SLASH.value:
        return "Slash the sixes."
    if action.value is EnumAction.FINISH_THREE_TIMES_SLASH.value:
        return "Slash the three times."
    if action.value is EnumAction.FINISH_FOUR_TIMES_SLASH.value:
        return "Slash the four times."
    if action.value is EnumAction.FINISH_FULL_HOUSE_SLASH.value:
        return "Slash the full house."
    if action.value is EnumAction.FINISH_SMALL_STREET_SLASH.value:
        return "Slash the small street."
    if action.value is EnumAction.FINISH_LARGE_STREET_SLASH.value:
        return "Slash the large street."
    if action.value is EnumAction.FINISH_KNIFFEL_SLASH.value:
        return "Slash the kniffel."
    if action.value is EnumAction.FINISH_CHANCE_SLASH.value:
        return "Slash the chance."


def print_action(action: EnumAction):
    print("        " + str(get_action(action)))


if __name__ == "__main__":
    clear = lambda: os.system("cls")
    yes_list = ["yes", "y", "YES", "Y"]

    clear()
    print_kniffel()
    print("")
    if input("    Do you want to play a new game? (y/n): ") in yes_list:
        clear()
        print_kniffel()
        print()
        print("    Started a new Kniffel Game!")
        print()
        rl = KniffelRL(
            load=False,
            config_path="src/config/config.csv",
            path_prefix="",
            hyperparater_base=None,
            env_observation_space=20,
            env_action_space=57,
        )

        env_config = {
            "reward_roll_dice": 0,
            "reward_game_over": -1000,
            "reward_finish": 150,
            "reward_bonus": 50,
        }

        agent = rl.build_use_agent(
            path="output/weights/p_date=2022-09-06-16_07_19",
            episodes=1,
            env_config=env_config,
            weights_name="weights_500000",
            logging=False,
        )

        if input("    Do you want to use your own dices? (y/n): ") in yes_list:
            kniffel = Kniffel(custom=True)
            clear()
            print_kniffel()
            while True:
                if (
                    kniffel.is_new_game()
                    or kniffel.get_last().status.value == KniffelStatus.INIT.value
                ):
                    print_header(kniffel, True)
                    print("    To start please input the dices you just rolled.")
                    dices = input(
                        "    Input your dices in the following style: 1 5 3 2 4: "
                    )
                    dices_list = [int(d) for d in dices.split(" ")]

                    kniffel.mock(DiceSet(mock=dices_list))

                    print_header(kniffel, False, True)
                elif kniffel.get_last().status.value == KniffelStatus.ATTEMPTING.value:
                    print_header(kniffel, False)

                action = agent.forward(kniffel.get_state())
                print()
                print("    The AI suggest that you do the following action:")
                print()
                print_action(EnumAction(action))
                print()

                if action <= 12 or action >= 44:
                    if (
                        input("    Do you want to accept this action? (y/n): ")
                        in yes_list
                    ):
                        rl.apply_prediction(kniffel, EnumAction(action))
                else:
                    while True:
                        if (
                            input(
                                "    Did you roll the dice as recommended by the AI? (y/n): "
                            )
                            in yes_list
                        ):
                            dices = input(
                                "    Input the new dices in the following style: 1 5 3 2 4: "
                            )
                            dices_list = [int(d) for d in dices.split(" ")]

                            kniffel.mock(DiceSet(mock=dices_list))

                            break
        else:
            kniffel = Kniffel()
            while True:
                state = kniffel.get_state()
                print()
                print(KniffelDraw().draw_dices(state[0][0:5]))
                print()
                print(KniffelDraw().draw_sheet(kniffel))

                action = agent.forward(state)
                enum_action = EnumAction(action)

                print(
                    f"    The AI suggest that you do the following action: {enum_action}"
                )

                if input("    Do you want to accept this action? (y/n): ") in yes_list:
                    rl.apply_prediction(kniffel, EnumAction(action))
                else:
                    new_action = input("    Give the id of the Action: ")
                    new_enum_action = EnumAction(int(new_action))

                    rl.apply_prediction(kniffel, EnumAction(new_enum_action))

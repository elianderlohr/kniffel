     __  ___ .__   __.  __   _______  _______  _______  __                    .______       __      
    |  |/  / |  \ |  | |  | |   ____||   ____||   ____||  |                   |   _  \     |  |     
    |  '  /  |   \|  | |  | |  |__   |  |__   |  |__   |  |         ______    |  |_)  |    |  |     
    |    <   |  . `  | |  | |   __|  |   __|  |   __|  |  |        |______|   |      /     |  |     
    |  .  \  |  |\   | |  | |  |     |  |     |  |____ |  `----.              |  |\  \----.|  `----.
    |__|\__\ |__| \__| |__| |__|     |__|     |_______||_______|              | _| `._____||_______|

The dice game "Kniffel", also known under the name "Yahtzee", coded in Python. 


More Information about this project can be viewed on my website: [https://kniffel.elch.cc/](https://kniffel.elch.cc/)

## The Game

The objective of the game is to score points by rolling five dice to make certain combinations. The dice can be rolled up to three times in a turn to try to make various scoring combinations and dice must remain in the box. A game consists of thirteen rounds. After each round, the player chooses which scoring category is to be used for that round. Once a category has been used in the game, it cannot be used again. The scoring categories have varying point values, some of which are fixed values and others for which the score depends on the value of the dice. A Yahtzee is five-of-a-kind and scores 50 points, the highest of any category. The winner is the player who scores the most points. [Source: Wikipedia/Yahtzee](https://en.wikipedia.org/wiki/Yahtzee)

## The Goal

The goal of this repo is to develop a deep reinforment learning model that manages to learn the rules of the game and score high-scores.

*Reinforcement Learning with deep neural network!*

# The Game
The game is developed in Python and is located unter */kniffel*. 

A sample usage of the game can be found unter [src/kniffel/game.py](src/kniffel/game.py). The game files are listed under [src/kniffel/classes](src/kniffel/classes/).

# The AI

The reinforcement learning logic is located under [src/rl](src/rl/).

## RL Description

### State

The current state from the kniffel game can be accessed by calling the get_state() function from the Kniffel object.

Example State:

    [[  0.66666667  0.33333333  0.83333333  0.5         0.33333333  0.33333333
        -1.         -1.         -1.         -1.         -1.         -1.
        0.          1.          1.          1.          1.          1.
        -1.         -1.        ]]

Each item in the list represents a parameter that describes the current state of the kniffel game.

State Explanation:

| Item Index | Description                                               | Value range |
|------------|-----------------------------------------------------------|-------------|
| 0   - 4    | Value of the 5 dices, scaled to   6                       | "0 - 1"       |
| 5          | Amount   of tries left for the current round, scaled to 3 | "0   - 1"     |
| 6          | Ones: -1 not logged, scaled to 5                          | "-1 to 1"   |
| 7          | Twos:   -1 not logged, scaled to 10                       | "-1   to 1" |
| 8          | Threes: -1 not logged, scaled to   15                     | "-1 to 1"   |
| 9          | Fours:   -1 not logged, scaled to 20                      | "-1   to 1" |
| 10         | Fives: -1 not logged, scaled to   25                      | "-1 to 1"   |
| 11         | Sixes:   -1 not logged, scaled to 30                      | "-1   to 1" |
| 12         | Has Bonus (1 bonus, 0 no bonus)                           | "0 - 1"       |
| 13         | Three   Times: -1 not logged, scaled to 30                | "-1   to 1" |
| 14         | Four Times: -1 not logged,   scaled to 30                 | "-1 to 1"   |
| 15         | Full   House: -1 not logged, scaled to 25                 | "-1   to 1" |
| 16         | Small Street: -1 not logged,   scaled to 30               | "-1 to 1"   |
| 17         | Large   Street: -1 not logged, scaled to 40               | "-1   to 1" |
| 18         | Kniffel: -1 not logged, scaled   to 50                    | "-1 to 1"   |
| 19         | Chance: -1 not logged, scaled to 30                       | "-1 to 1"   |

### Action

The kniffel RL can be controlled by calling 57 individual actions. Each runns a special command. The action with the ID 2 for example finishes the current turn and logs the threes.

To ensure no mixup with the action id, all actions are easy to access using the [EnumAction](src/rl/env.py) enum class.


    # Finish Actions
    FINISH_ONES = "Finish the ones."
    FINISH_TWOS = "Finish the two."
    FINISH_THREES = "Finish the threes."
    FINISH_FOURS = "Finish the fours."
    FINISH_FIVES = "Finish the fives."
    FINISH_SIXES = "Finish the sixes."
    FINISH_THREE_TIMES = "Finish the three times."
    FINISH_FOUR_TIMES = "Finish the four times."
    FINISH_FULL_HOUSE = "Finish the full house."
    FINISH_SMALL_STREET = "Finish the small street."
    FINISH_LARGE_STREET = "Finish the large street."
    FINISH_KNIFFEL = "Finish the kniffel."
    FINISH_CHANCE = "Finish the chance."

    # Continue Actions
    NEXT_0 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 0, 0"
    NEXT_1 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 0, 1"
    NEXT_2 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 1, 0"
    NEXT_3 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 0, 1, 1"
    NEXT_4 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 0, 0"
    NEXT_5 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 0, 1"
    NEXT_6 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 1, 0"
    NEXT_7 = "Roll the dices (0 = re-roll, 1 = keep): 0, 0, 1, 1, 1"
    NEXT_8 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 0, 0"
    NEXT_9 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 0, 1"
    NEXT_10 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 1, 0"
    NEXT_11 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 0, 1, 1"
    NEXT_12 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 0, 0"
    NEXT_13 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 0, 1"
    NEXT_14 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 1, 0"
    NEXT_15 = "Roll the dices (0 = re-roll, 1 = keep): 0, 1, 1, 1, 1"
    NEXT_16 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 0, 0"
    NEXT_17 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 0, 1"
    NEXT_18 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 1, 0"
    NEXT_19 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 0, 1, 1"
    NEXT_20 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 0, 0"
    NEXT_21 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 0, 1"
    NEXT_22 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 1, 0"
    NEXT_23 = "Roll the dices (0 = re-roll, 1 = keep): 1, 0, 1, 1, 1"
    NEXT_24 = "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 0, 0"
    NEXT_25 = "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 0, 1"
    NEXT_26 = "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 1, 0"
    NEXT_27 = "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 0, 1, 1"
    NEXT_28 = "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 1, 0, 0"
    NEXT_29 = "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 1, 0, 1"
    NEXT_30 = "Roll the dices (0 = re-roll, 1 = keep): 1, 1, 1, 1, 0"

    # FINISH Actions
    FINISH_ONES_SLASH = "Slash the ones."
    FINISH_TWOS_SLASH = "Slash the two."
    FINISH_THREES_SLASH = "Slash the threes."
    FINISH_FOURS_SLASH = "Slash the fours."
    FINISH_FIVES_SLASH = "Slash the fives."
    FINISH_SIXES_SLASH = "Slash the sixes."
    FINISH_THREE_TIMES_SLASH = "Slash the three times."
    FINISH_FOUR_TIMES_SLASH = "Slash the four times."
    FINISH_FULL_HOUSE_SLASH = "Slash the full house."
    FINISH_SMALL_STREET_SLASH = "Slash the small street."
    FINISH_LARGE_STREET_SLASH = "Slash the large street."
    FINISH_KNIFFEL_SLASH = "Slash the kniffel."
    FINISH_CHANCE_SLASH = "Slash the chance."

## Use the AI model

The AI model can be used by running the game.py script in the src/console folder. This starts a simple bash based command prompt wich allows to play a kniffel game.

![Example Game](img/example.gif)

## Highscores

| Name / ID | Date       | Training Episodes | Duration | AVG Score | AVG Rounds | Weights                            |
|----|------------|-------------------|----------|-------|-|------------------------------------|
| model_4  | 21.08.2022 | 20.000.000+         | ~3d       | ~115   | ~26 | [output/weights/model_5](output/weights/model_4) |

*tbc*

## Example Game

    ####################################################################################
    ##  Try: 27
    ##  Action: EnumAction.NEXT_16

    -----     -----     -----     -----     -----     
    |o o|     |o  |     |o  |     |o o|     |o  |     
    |o o|     |   |     | o |     |   |     |   |     
    |o o|     |  o|     |  o|     |o o|     |  o|     
    -----     -----     -----     -----     -----     

     ____________________________________      ____________________________________
    | CATEGORY       | STATE  |  POINTS  |    | CATEGORY       | STATE  |  POINTS  |
    |::::::::::::::::|::::::::|::::::::::|    |::::::::::::::::|::::::::|::::::::::|
    | ONES           |    1   |     2    |    | THREE TIMES    |    1   |    11    |
    |================|========|==========|    |================|========|==========|
    | TWOS           |    1   |     4    |    | FOURS TIMES    |   -1   |     0    |
    |================|========|==========|    |================|========|==========|
    | THREES         |    1   |     9    |    | FULL HOUSE     |   -1   |     0    |
    |================|========|==========|    |================|========|==========|
    | FOURS          |    1   |     4    |    | SMALL STREET   |   -1   |     0    |
    |================|========|==========|    |================|========|==========|
    | FIVES          |    1   |    15    |    | LARGE STREET   |   -1   |     0    |
    |================|========|==========|    |================|========|==========|
    | SIXES          |    1   |    12    |    | KNIFFEL        |   -1   |     0    |
    |################|########|##########|    |================|========|==========|
    |  BONUS         |    0   |     0    |    | CHANCE         |    1   |    24    |
    |================|========|==========|    |################|########|##########|
    |  TOP POINTS    |        |    46    |    |  BOTTOM POINTS |        |    35    |
    |################|########|##########|    |================|========|==========|
                                              |  TOTAL POINTS  |        |    81    |
                                              |################|########|##########|


Example Game Log: [Example Game Log](output/weights/model_4/game_log/log.txt)

## Hyperparameter optimization
To optimize the hyperparameter selection the library optuna is used. The file is located under [src/rl/optuna.py](src/rl/optuna.py).

_Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters._

[Optuna: A hyperparameter optimization framework](https://github.com/optuna/optuna#optuna-a-hyperparameter-optimization-framework)

### Commands

#### Start Hyperparameter optimization with optuna

```shell
python3 src/rl/optuna.py --pw XXX --study_name kniffel_v09.05.1 --new true --jobs 10
```

#### Start Optuna Dashboard

```shell
optuna-dashboard mysql+pymysql://kniffel:XXX@kniffel-do-user-12010256-0.b.db.ondigitalocean.com:25060/kniffel
```
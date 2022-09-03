     __  ___ .__   __.  __   _______  _______  _______  __          ___       __  
    |  |/  / |  \ |  | |  | |   ____||   ____||   ____||  |        /   \     |  | 
    |  '  /  |   \|  | |  | |  |__   |  |__   |  |__   |  |       /  ^  \    |  | 
    |    <   |  . `  | |  | |   __|  |   __|  |   __|  |  |      /  /_\  \   |  | 
    |  .  \  |  |\   | |  | |  |     |  |     |  |____ |  |____ /  _____  \  |  | 
    |__|\__\ |__| \__| |__| |__|     |__|     |_______||_______/__/     \__\ |__| 
                                                                              

The dice game "Kniffel", also known under the name "Yahtzee", coded in Python. 

## The Game

The objective of the game is to score points by rolling five dice to make certain combinations. The dice can be rolled up to three times in a turn to try to make various scoring combinations and dice must remain in the box. A game consists of thirteen rounds. After each round, the player chooses which scoring category is to be used for that round. Once a category has been used in the game, it cannot be used again. The scoring categories have varying point values, some of which are fixed values and others for which the score depends on the value of the dice. A Yahtzee is five-of-a-kind and scores 50 points, the highest of any category. The winner is the player who scores the most points. [Source: Wikipedia/Yahtzee](https://en.wikipedia.org/wiki/Yahtzee)

## The Goal

The goal of this repo is to develop a deep reinforment learning model that manages to learn the rules of the game and score high-scores.

*Reinforcement Learning with deep neural network!*

# The Game
The game is developed in Python and is located unter */kniffel*. 

A sample usage of the game can be found unter [src/kniffel/game.py](src/kniffel/game.py). The game files are listed under [src/kniffel/classes](src/kniffel/classes/).

# The AI

The reinforcement learning logic is located under [src/ai](src/ai/).

## Highscores

| Name / ID | Date       | Training Episodes | Duration | AVG Score | AVG Rounds | Weights                            |
|----|------------|-------------------|----------|-------|-|------------------------------------|
| model_4  | 21.08.2022 | 20.000.000+         | ~3d       | ~115   | ~26 | [output/weights/model_4](output/weights/model_4) |

*tbc*

## Hyperparameter optimization
To optimize the hyperparameter selection the library optuna is used. The file is located under [src/ai/ai_optuna.py](src/ai/ai_optuna.py).

_Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. It features an imperative, define-by-run style user API. Thanks to our define-by-run API, the code written with Optuna enjoys high modularity, and the user of Optuna can dynamically construct the search spaces for the hyperparameters._

[Optuna: A hyperparameter optimization framework](https://github.com/optuna/optuna#optuna-a-hyperparameter-optimization-framework)


### Commands

#### Start Hyperparameter optimization with optuna

```shell
python src/ai/ai_optuna.py --pw XXX --study_name kniffel_22_08_16_m1 --new true --jobs 6
```

#### Start Optuna Dashboard

```shell
optuna-dashboard mysql+pymysql://kniffel:XXX@kniffel-do-user-12010256-0.b.db.ondigitalocean.com:25060/kniffel
```
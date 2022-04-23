# KniffelAI
The dice game "Kniffel", also known under the name "Yahtzee", coded in Python. 

## The Game

The objective of the game is to score points by rolling five dice to make certain combinations. The dice can be rolled up to three times in a turn to try to make various scoring combinations and dice must remain in the box. A game consists of thirteen rounds. After each round, the player chooses which scoring category is to be used for that round. Once a category has been used in the game, it cannot be used again. The scoring categories have varying point values, some of which are fixed values and others for which the score depends on the value of the dice. A Yahtzee is five-of-a-kind and scores 50 points, the highest of any category. The winner is the player who scores the most points. [Source: Wikipedia/Yahtzee](https://en.wikipedia.org/wiki/Yahtzee)

## The Goal

The goal of this repo is to develop a deep neural network that manages to learn the rules of the game and score high-scores. The agent learns the game by getting rewarded or penaltizes. 

*Reinforcement Learning with deep neural network!*

# The Game
The game is developed in Python and is located unter */kniffel*. 

A sample usage of the game can be found unter [kniffel/game.py](kniffel/game.py). The game files are listed under [kniffel/classes](kniffel/classes/).

Currently there is **NO** Jupyter notebook support.

# The AI

The neural network is located under [ai/ai.py](ai/ai.py).

## Highscores

| Id | Date       | Training Episodes | Duration | AVG Score | AVG Rounds | Weights                            |
|----|------------|-------------------|----------|-------|-|------------------------------------|
| 1  | 23.04.2022 | 500.000         | 4.5h       | 27.2   | 8.7 | [weights\p_date=2022-04-23-14_24_28](weights\p_date=2022-04-23-14_24_28) |
*tbc*
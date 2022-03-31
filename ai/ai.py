from cgi import test
from cgitb import html
from pprint import pprint
from tkinter import E
from env import KniffelEnv
from datetime import datetime

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf

import numpy as np
import os
import sys
import inspect

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    ),
)

from kniffel.classes.options import KniffelOptions
from kniffel.classes.kniffel import Kniffel
from env import EnumAction


def build_model(actions, windows_length):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(windows_length, 13, 16)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))

    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions, windows_length):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=100000, window_length=windows_length)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-3,
        batch_size=64,
    )
    return dqn


def do_random():
    env = KniffelEnv()

    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
            print(n_state)

        print("Episode:{} Score:{}".format(episode, score))


def train():
    env = KniffelEnv()

    states = env.observation_space.shape
    actions = env.action_space.n
    windows_length = 1

    model = build_model(actions, windows_length)
    dqn = build_agent(model, actions, windows_length)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])
    dqn.fit(env, nb_steps=1000000, visualize=False, verbose=1)

    scores = dqn.test(env, nb_episodes=100, visualize=False)
    print(np.mean(scores.history["episode_reward"]))
    _ = dqn.test(env, nb_episodes=15, visualize=False)

    date = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    name = (
        "dqn_weights_"
        + str(date)
        + "__"
        + str(np.mean(scores.history["episode_reward"]))
        + "__.h5f"
    )
    print(name)
    dqn.save_weights("weights/" + name, overwrite=True)


def predict(dqn: DQNAgent, kniffel: Kniffel, state):
    action = dqn.forward(state)
    enum_action = EnumAction(action)

    # print("Based on state predicted: " + str(enum_action))

    if EnumAction.FINISH_ONES is enum_action:
        kniffel.finish_turn(KniffelOptions.ONES)
    if EnumAction.FINISH_TWOS is enum_action:
        kniffel.finish_turn(KniffelOptions.TWOS)
    if EnumAction.FINISH_THREES is enum_action:
        kniffel.finish_turn(KniffelOptions.THREES)
    if EnumAction.FINISH_FOURS is enum_action:
        kniffel.finish_turn(KniffelOptions.FOURS)
    if EnumAction.FINISH_FIVES is enum_action:
        kniffel.finish_turn(KniffelOptions.FIVES)
    if EnumAction.FINISH_SIXES is enum_action:
        kniffel.finish_turn(KniffelOptions.SIXES)
    if EnumAction.FINISH_THREE_TIMES is enum_action:
        kniffel.finish_turn(KniffelOptions.THREE_TIMES)
    if EnumAction.FINISH_FOUR_TIMES is enum_action:
        kniffel.finish_turn(KniffelOptions.FOUR_TIMES)
    if EnumAction.FINISH_FULL_HOUSE is enum_action:
        kniffel.finish_turn(KniffelOptions.FULL_HOUSE)
    if EnumAction.FINISH_SMALL_STREET is enum_action:
        kniffel.finish_turn(KniffelOptions.SMALL_STREET)
    if EnumAction.FINISH_LARGE_STREET is enum_action:
        kniffel.finish_turn(KniffelOptions.LARGE_STREET)
    if EnumAction.FINISH_KNIFFEL is enum_action:
        kniffel.finish_turn(KniffelOptions.KNIFFEL)
    if EnumAction.FINISH_CHANCE is enum_action:
        kniffel.finish_turn(KniffelOptions.CHANCE)

    # Continue enum_actions
    if EnumAction.NEXT_0 is enum_action:
        kniffel.add_turn(keep=[0, 0, 0, 0, 0])
    if EnumAction.NEXT_1 is enum_action:
        kniffel.add_turn(keep=[0, 0, 0, 0, 1])
    if EnumAction.NEXT_2 is enum_action:
        kniffel.add_turn(keep=[0, 0, 0, 1, 0])
    if EnumAction.NEXT_3 is enum_action:
        kniffel.add_turn(keep=[0, 0, 0, 1, 1])
    if EnumAction.NEXT_4 is enum_action:
        kniffel.add_turn(keep=[0, 0, 1, 0, 0])
    if EnumAction.NEXT_5 is enum_action:
        kniffel.add_turn(keep=[0, 0, 1, 0, 1])
    if EnumAction.NEXT_6 is enum_action:
        kniffel.add_turn(keep=[0, 0, 1, 1, 0])
    if EnumAction.NEXT_7 is enum_action:
        kniffel.add_turn(keep=[0, 0, 1, 1, 1])
    if EnumAction.NEXT_8 is enum_action:
        kniffel.add_turn(keep=[0, 1, 0, 0, 0])
    if EnumAction.NEXT_9 is enum_action:
        kniffel.add_turn(keep=[0, 1, 0, 0, 1])
    if EnumAction.NEXT_10 is enum_action:
        kniffel.add_turn(keep=[0, 1, 0, 1, 0])
    if EnumAction.NEXT_11 is enum_action:
        kniffel.add_turn(keep=[0, 1, 0, 1, 1])
    if EnumAction.NEXT_12 is enum_action:
        kniffel.add_turn(keep=[0, 1, 1, 0, 0])
    if EnumAction.NEXT_13 is enum_action:
        kniffel.add_turn(keep=[0, 1, 1, 0, 1])
    if EnumAction.NEXT_14 is enum_action:
        kniffel.add_turn(keep=[0, 1, 1, 1, 0])
    if EnumAction.NEXT_15 is enum_action:
        kniffel.add_turn(keep=[0, 1, 1, 1, 1])
    if EnumAction.NEXT_16 is enum_action:
        kniffel.add_turn(keep=[1, 0, 0, 0, 0])
    if EnumAction.NEXT_17 is enum_action:
        kniffel.add_turn(keep=[1, 0, 0, 0, 1])
    if EnumAction.NEXT_18 is enum_action:
        kniffel.add_turn(keep=[1, 0, 0, 1, 0])
    if EnumAction.NEXT_19 is enum_action:
        kniffel.add_turn(keep=[1, 0, 0, 1, 1])
    if EnumAction.NEXT_20 is enum_action:
        kniffel.add_turn(keep=[1, 0, 1, 0, 0])
    if EnumAction.NEXT_21 is enum_action:
        kniffel.add_turn(keep=[1, 0, 1, 0, 1])
    if EnumAction.NEXT_22 is enum_action:
        kniffel.add_turn(keep=[1, 0, 1, 1, 0])
    if EnumAction.NEXT_23 is enum_action:
        kniffel.add_turn(keep=[1, 0, 1, 1, 1])
    if EnumAction.NEXT_24 is enum_action:
        kniffel.add_turn(keep=[1, 1, 0, 0, 0])
    if EnumAction.NEXT_25 is enum_action:
        kniffel.add_turn(keep=[1, 1, 0, 0, 1])
    if EnumAction.NEXT_26 is enum_action:
        kniffel.add_turn(keep=[1, 1, 0, 1, 0])
    if EnumAction.NEXT_27 is enum_action:
        kniffel.add_turn(keep=[1, 1, 0, 1, 1])
    if EnumAction.NEXT_28 is enum_action:
        kniffel.add_turn(keep=[1, 1, 1, 0, 0])
    if EnumAction.NEXT_29 is enum_action:
        kniffel.add_turn(keep=[1, 1, 1, 0, 1])
    if EnumAction.NEXT_30 is enum_action:
        kniffel.add_turn(keep=[1, 1, 1, 1, 0])
    if EnumAction.NEXT_31 is enum_action:
        kniffel.add_turn(keep=[1, 1, 1, 1, 1])


def use():
    env = KniffelEnv()

    actions = env.action_space.n

    model = build_model(actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])

    dqn.load_weights("weights/dqn_weights_2022_03_31-11_47_48_AM__-219.8__.h5f")
    # dqn.load_weights("weights/dqn_weights.h5f")

    points = []
    break_counter = 0
    n = 1000
    for i in range(n):
        kniffel = Kniffel()
        while True:
            try:
                state = kniffel.get_array()
                predict(dqn, kniffel, state)
            except Exception as e:
                points.append(kniffel.get_points())
                break_counter += 1
                break

            points.append(kniffel.get_points())

    print(f"Break counter: {break_counter}/{n}")
    print("Avg points: " + str(sum(points) / len(points)))
    print("Max points: " + str(max(points)))
    print("Min points: " + str(min(points)))


if __name__ == "__main__":
    train()

from env import KniffelEnv
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf

from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=states))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2,
    )
    return dqn


def do_random(env):
    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        print("Reset Env")

        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward

        print(state)
        print("Episode:{} Score:{}".format(episode, score))


def main(env):
    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(states, actions)

    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

    # scores = dqn.test(env, nb_episodes=100, visualize=False)
    # print(np.mean(scores.history["episode_reward"]))
    # _ = dqn.test(env, nb_episodes=15, visualize=True)


if __name__ == "__main__":
    env = KniffelEnv()

    main(env)

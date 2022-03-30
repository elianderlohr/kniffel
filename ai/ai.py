from env import KniffelEnv

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

import tensorflow as tf

import numpy as np


def build_model(states, actions):
    model = tf.keras.Sequential()
    model.add(Flatten(input_shape=(1, 13, 16)))
    model.add(Dense(265, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))

    model.add(Dense(actions, activation="linear"))
    return model


def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = DQNAgent(
        model=model,
        memory=memory,
        policy=policy,
        nb_actions=actions,
        nb_steps_warmup=10,
        target_model_update=1e-2,
    )
    return dqn


def do_random():
    env = KniffelEnv()

    episodes = 10
    for episode in range(1, episodes + 1):
        state = env.reset()
        done = False
        score = 0

        print()

        while not done:
            # env.render()
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward

        print("Episode:{} Score:{}".format(episode, score))


def main():
    env = KniffelEnv()

    states = env.observation_space.shape
    actions = env.action_space.n

    model = build_model(states, actions)
    dqn = build_agent(model, actions)
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])
    dqn.fit(env, nb_steps=500000, visualize=False, verbose=1)

    scores = dqn.test(env, nb_episodes=100, visualize=False)
    print(np.mean(scores.history["episode_reward"]))
    _ = dqn.test(env, nb_episodes=15, visualize=False)

    dqn.save_weights("dqn_weights.h5f", overwrite=True)


if __name__ == "__main__":
    main()

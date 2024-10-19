import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(episodes, is_training=True, render=False):

    env = gym.make('Taxi-v3', render_mode='human' if render else None)

    if is_training:
        q = np.zeros((env.observation_space.n, env.action_space.n))  # init a 500 x 6 array
    else:
        with open('taxi.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.9  # alpha or learning rate
    discount_factor_g = 0.9  # gamma or discount rate
    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.0001  # epsilon decay rate
    rng = np.random.default_rng()  # random number generator

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 63
        terminated = False  # True when fall in hole or reached goal
        truncated = False  # True when actions > 200

        rewards = 0
        while not terminated and not truncated:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()  # actions: 0=left, 1=down, 2=right, 3=up
            else:
                action = np.argmax(q[state, :])

            new_state, reward, terminated, truncated, _ = env.step(action)

            rewards += reward

            if is_training:
                q[state, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state, :]) - q[state, action]
                )

            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if epsilon == 0:
            learning_rate_a = 0.0001

        rewards_per_episode[i] = rewards

        # Print rewards for training every 1000 episodes
        if is_training and (i + 1) % 1000 == 0:
            print(f"Rewards after {i + 1} episodes: {np.sum(rewards_per_episode[max(0, i - 999):(i + 1)])/1000}")

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t - 100):(t + 1)])
    plt.plot(sum_rewards)
    plt.savefig('taxi.png')

    if is_training:
        with open("taxi.pkl", "wb") as f:
            pickle.dump(q, f)

    # Print rewards per episode in a table format for the testing phase
    if not is_training:
        print("\nRewards per Episode:")
        print(f"{'Episode':<10} {'Reward':<10}")
        print("-" * 20)
        for episode in range(episodes):
            print(f"{episode + 1:<10} {rewards_per_episode[episode]:<10}")

if __name__ == '__main__':
    # Training for 15000 episodes
    run(15000)
    # Showing output for 10 episodes after training
    run(10, is_training=False, render=True)

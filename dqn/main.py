import gym
import torch
import numpy as np
from collections import deque
import random
import pygame
import time
import matplotlib.pyplot as plt

# Assuming your DQN model, training, and utility functions are defined in the same script for simplicity
from DQN_model import DQN, train, select_action
from utils import preprocess_state, plot_rewards
import env_gib9170  # assuming your custom environment module

# Function to render the environment
def render(env, screen):
    screen.blit(env.background_image, (0, 0))
    for cat_state in env.cat_states:
        screen.blit(env.cat_image, (cat_state[0] * env.square_size, cat_state[1] * env.square_size))
    screen.blit(env.goal_image, (env.goal_coordinates[0] * env.square_size, env.goal_coordinates[1] * env.square_size))
    screen.blit(env.mouse_image, (env.mouse_state[0] * env.square_size, env.mouse_state[1] * env.square_size))
    pygame.display.flip()

# Function to train the DQN model
def train_dqn(env, model, optimizer, criterion, screen, n_episodes=5000, max_steps=100, gamma=0.99,
              epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64, replay_buffer_size=10000):
    epsilon = epsilon_start
    replay_buffer = deque(maxlen=replay_buffer_size)
    rewards = []
    

    for episode in range(n_episodes):
        state = env.reset()
        state = preprocess_state(state)
        episode_reward = 0

        for step in range(max_steps):
            action = select_action(model, state, epsilon, env.action_space.n)
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            episode_reward += reward

            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state

            # Render the environment (optional, uncomment for visualization)
            #render(env, screen)
            #time.sleep(0.1)  # Add a small delay to better observe the rendering

            if done:
                break

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_batch, next_states, dones = zip(*batch)
                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
                rewards_batch = torch.tensor(rewards_batch, dtype=torch.float32).unsqueeze(1)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

                q_values = model(states).gather(1, actions)
                next_q_values = model(next_states).max(1)[0].unsqueeze(1)
                targets = rewards_batch + gamma * next_q_values * (1 - dones)

                loss = train(model, optimizer, criterion, states, targets)

        rewards.append(episode_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(f"Episode {episode + 1}, Reward: {episode_reward}, Epsilon: {epsilon:.2f}")

        # Save the model and plot at intervals
        if (episode + 1) % 5000 == 0:
            torch.save(model.state_dict(), f"dqn_.pth")
            plot_rewards(rewards, episode + 1)

    return rewards

# Function to plot rewards
def plot_rewards(rewards, episode):
    plt.figure()
    plt.plot(rewards, label="Reward per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.legend()
    plt.title(f'Rewards up to Episode {episode}')
    plt.savefig(f'rewards_plot_{episode}.png')
    plt.show()

# Function to test the trained DQN model
def test(model, env, screen, max_steps=100):
    total_rewards = []

    for episode in range(10):  # Test 10 episodes
        state = env.reset()
        state = preprocess_state(state)
        episode_reward = 0

        for step in range(max_steps):
            action = select_action(model, state, epsilon=0.0, n_actions=env.action_space.n)  # Greedy action selection
            next_state, reward, done, _ = env.step(action)
            next_state = preprocess_state(next_state)
            episode_reward += reward

            # Render the environment
            #render(env, screen)
            #time.sleep(0.1)  # Add a small delay to better observe the rendering

            if done:
                break

            state = next_state

        total_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}, Reward: {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"Average Test Reward: {avg_reward}")

    return avg_reward, total_rewards

# Main function to orchestrate training, testing, and rendering
def main():
    # Initialize the environment
    env = env_gib9170.CatAndMouseEnv(grid_size=15)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    # Initialize the DQN model, optimizer, and loss criterion
    model = DQN(state_dim, n_actions)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()

    # Define training parameters
    n_episodes = 5000
    max_steps = 100
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    batch_size = 64

    # Initialize Pygame
    pygame.init()
    pygame.mixer.init()

    # Set up the display
    screen = pygame.display.set_mode((env.window_width, env.window_height))
    pygame.display.set_caption("Cat and Mouse Game")

    # Train the DQN model
    train_rewards = train_dqn(env, model, optimizer, criterion, screen, n_episodes, max_steps, gamma,
                              epsilon_start, epsilon_decay, epsilon_min, batch_size)

    # Test the trained model
    test_avg_reward, test_rewards = test(model, env, screen, max_steps)

    # Plot training rewards
    plt.figure()
    plt.plot(train_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.savefig('training_curve.png')  # Save the figure
    plt.show()

    # Plot test rewards
    plt.figure()
    plt.plot(test_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Test Rewards')
    plt.savefig('dqn.pth')  # Save the figure
    plt.show()

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()

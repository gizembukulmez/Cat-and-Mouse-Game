import numpy as np
import matplotlib.pyplot as plt

def preprocess_state(state):
    return np.array(state, dtype=np.float32)

def plot_rewards(rewards):
    plt.figure()
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.show()

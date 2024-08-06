# Imports:
# --------
from padm_env import create_env
from Q_learning import train_q_learning, visualize_q_table

# User definitions:
# -----------------
train = True #
visualize_results = True

learning_rate = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 0.9  # Exploration rate
epsilon_min = 1.0  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for exploration
no_episodes = 1000  # Number of episodes

goal_coordinates = (11, 11)
# Define all hell state coordinates as a tuple within a list
cat_states= [(3, 3), (6, 6), (8, 8)]


# Function to create an instance of the environment and visualize it
# ------------------------------------------------------------------
def main():
    # Create an instance of the environment:
    # --------------------------------------
    env = create_env(goal_coordinates=goal_coordinates,
                     cat_states=cat_states)

    if train:
        # Train a Q-learning agent:
        # -------------------------
        train_q_learning(env=env,
                         no_episodes=no_episodes,
                         epsilon=epsilon,
                         epsilon_min=epsilon_min,
                         epsilon_decay=epsilon_decay,
                         alpha=learning_rate,
                         gamma=gamma)

    if visualize_results:
        # Visualize the Q-table:
        # ----------------------
        visualize_q_table(cat_states=cat_states,
                          goal_coordinates=goal_coordinates,
                          q_values_path="q_table.npy")

    # Close the environment when done:
    env.close()


# Entry point of the program
if __name__ == "__main__":
    main()

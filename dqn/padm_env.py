# Imports:
# --------
import gymnasium as gym
import numpy as np
import pygame
import sys


# Class 1: Define a custom environment
# --------

class CatAndMouseEnv(gym.Env):
    def __init__(self, grid_size=11, goal_coordinates=None, cat_states=None):
    #def __init__(self, grid_size=10, goal_coordinates=(10, 10), cat_states=None):
    #def __init__(self, grid_size=10, goal_coordinates=(10, 10), cat_states=[(2, 2), (5, 5), (7, 7)]): #[np.array([2, 2]), np.array([5, 5]), np.array([7, 7])]): #
        super(CatAndMouseEnv, self).__init__()
        self.grid_size = grid_size
        self.cell_size = 50
        self.state = None
        self.reward = 0
        self.info = {}
        self.goal = np.array(goal_coordinates if goal_coordinates is not None else [self.grid_size - 1, self.grid_size - 1])
        self.done = False
        self.cat_states = [np.array(cat) for cat in cat_states] if cat_states else [np.array([2, 2]), np.array([5, 5]), np.array([7, 7])] # Hell states

        # Action-space:
        self.action_space = gym.spaces.Discrete(4)

        # Observation space:
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32)

        # Initialize the window:
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        pygame.display.set_caption("Cat and Mouse Game")

        # Load pictures
        self.background_image = pygame.image.load("background.png").convert()
        self.background_image = pygame.transform.scale(self.background_image, (self.cell_size*self.grid_size, self.cell_size*self.grid_size))
        self.mouse_image = pygame.image.load("mouse.png").convert_alpha()
        self.mouse_image = pygame.transform.scale(self.mouse_image, (self.cell_size, self.cell_size))
        self.cat_image = pygame.image.load("cat.png").convert_alpha()
        self.cat_image = pygame.transform.scale(self.cat_image, (self.cell_size, self.cell_size))
        self.goal_image = pygame.image.load("door.jpeg").convert_alpha()
        self.goal_image = pygame.transform.scale(self.goal_image, (self.cell_size, self.cell_size))

        # Upload voice effect
        self.tom_and_jerry_sound = pygame.mixer.Sound("tomandjerry.mp3")
        self.music = True


    # Method 1: .reset()
    # ---------

    def reset(self):
        """
        Everything must be reset
        """
        self.state = np.array([0, 0])
        self.done = False
        self.reward = 0

        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )

        return self.state, self.info

    # Method 2: Add hell states
    # ---------

    def add_cat_states(self, cat_states):
        self.cat_states.append(np.array(cat_states))

    # Method 3: .step()
    # ---------

    def step(self, action):
        # Actions:
        # --------
        # Up:
        if action == 0 and self.state[0] > 0:
            self.state[0] -= 1

        # Down:
        if action == 1 and self.state[0] < self.grid_size-1:
            self.state[0] += 1

        # Right:
        if action == 2 and self.state[1] < self.grid_size-1:
            self.state[1] += 1

        # Left:
        if action == 3 and self.state[1] > 0:
            self.state[1] -= 1

        # Reward:
        # -------



        if np.array_equal(self.state, self.goal):  # Check goal condition
            self.reward = +1000
            self.done = True
        # Check hell-states
        elif True in [np.array_equal(self.state, cat) for cat in self.cat_states]:
            self.reward += -500
            self.done = True
        else:  # Every other state
            self.reward -= 0.1
            self.done = False

        # Info:
        # -----
        self.info["Distance to goal"] = np.sqrt(
            (self.state[0]-self.goal[0])**2 +
            (self.state[1]-self.goal[1])**2
        )

        return self.state, self.reward, self.done, self.info

    # Method 3: .render()
    # ---------

    def render(self):
    # Handle closing the window:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Clear the screen with white:
        self.screen.fill((255, 255, 255))

        # Draw background image:
        self.screen.blit(self.background_image, (0, 0))

        # Draw grid lines:
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                grid_rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.screen, (200, 200, 200), grid_rect, 1)

        # Draw the goal state:
        goal_rect = pygame.Rect(self.goal[1] * self.cell_size, self.goal[0] * self.cell_size, self.cell_size, self.cell_size)
        self.screen.blit(self.goal_image, goal_rect)

        # Draw the cat states:
        for cat in self.cat_states:
            cat_rect = pygame.Rect(cat[1] * self.cell_size, cat[0] * self.cell_size, self.cell_size, self.cell_size)
            self.screen.blit(self.cat_image, cat_rect)

        # Draw the agent (mouse):
        agent_rect = pygame.Rect(self.state[1] * self.cell_size, self.state[0] * self.cell_size, self.cell_size, self.cell_size)
        self.screen.blit(self.mouse_image, agent_rect)

        if self.music:
            self.tom_and_jerry_sound.play()
            self.music = False

        # Update the display:
        pygame.display.flip()


    # Method 4: .close()
    # ---------

    def close(self):
        pygame.quit()


# Function 1: Create an instance of the environment
# -----------
def create_env(goal_coordinates, cat_states):
    # Create the environment:
    env = CatAndMouseEnv(goal_coordinates=goal_coordinates, cat_states=cat_states)
    return env
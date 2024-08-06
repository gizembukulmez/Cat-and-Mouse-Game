import gym
import pygame
import numpy as np

class CatAndMouseEnv(gym.Env):
    def __init__(self, grid_size=15):
        super().__init__()
        self.grid_size = grid_size
        self.init_state = np.array([0, 0])
        self.mouse_state = self.init_state.copy()
        self.goal_coordinates = np.array([self.grid_size - 1, self.grid_size - 1])
        self.cat_states = [np.array([2, 2]), np.array([5, 5]), np.array([7, 7])]
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=self.grid_size-1, shape=(2,), dtype=np.int32)
        self.square_size = 50
        self.window_width = self.grid_size * self.square_size
        self.window_height = self.grid_size * self.square_size
        self.music = True

        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Cat and Mouse Game")

        # Load pictures
        self.background_image = pygame.image.load("background.png").convert()
        self.background_image = pygame.transform.scale(self.background_image, (self.window_width, self.window_height))
        self.mouse_image = pygame.image.load("mouse.png").convert_alpha()
        self.mouse_image = pygame.transform.scale(self.mouse_image, (self.square_size, self.square_size))
        self.cat_image = pygame.image.load("cat.png").convert_alpha()
        self.cat_image = pygame.transform.scale(self.cat_image, (self.square_size, self.square_size))
        self.goal_image = pygame.image.load("door.jpeg").convert_alpha()
        self.goal_image = pygame.transform.scale(self.goal_image, (self.square_size, self.square_size))

        # Upload voice effect
        self.tom_and_jerry_sound = pygame.mixer.Sound("/Users/afragizembukulmez/Desktop/deneme/tomandjerry.mp3")

    def reset(self):
        self.mouse_state = self.init_state.copy()
        return self.mouse_state

    def step(self, action):
        if action == 0:  # Move up
            self.mouse_state[1] = max(self.mouse_state[1] - 1, 0)
        elif action == 1:  # Move down
            self.mouse_state[1] = min(self.mouse_state[1] + 1, self.grid_size - 1)
        elif action == 2:  # Move left
            self.mouse_state[0] = max(self.mouse_state[0] - 1, 0)
        elif action == 3:  # Move right
            self.mouse_state[0] = min(self.mouse_state[0] + 1, self.grid_size - 1)

        reward = -1  # Small negative reward to encourage shortest path
        done = False

        if any(np.array_equal(self.mouse_state, cat_state) for cat_state in self.cat_states):
            reward = -100
            done = True
        elif np.array_equal(self.mouse_state, self.goal_coordinates):
            reward = 1500  # High positive reward for reaching the goal
            done = True

        return self.mouse_state, reward, done, {}

    def render(self):
        if self.music:
            self.tom_and_jerry_sound.play()
            self.music = False

        # Clear the screen with the background image
        self.screen.blit(self.background_image, (0, 0))

        # Draw the cats on the screen
        for cat_state in self.cat_states:
            self.screen.blit(self.cat_image, (cat_state[0] * self.square_size, cat_state[1] * self.square_size))

        # Draw the escape gate
        self.screen.blit(self.goal_image, (self.goal_coordinates[0] * self.square_size, self.goal_coordinates[1] * self.square_size))
        
        # Draw the mouse
        self.screen.blit(self.mouse_image, (self.mouse_state[0] * self.square_size, self.mouse_state[1] * self.square_size))

        # Update the display to show the new positions of objects
        pygame.display.flip()

    def close(self):
        pygame.quit()

if __name__ == "__main__":
    env = CatAndMouseEnv(grid_size=15)  # Grid size can be adjusted as needed
    state = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        action = np.random.randint(4)  # Random move for the mouse
        state, reward, done, _ = env.step(action)
        print(f"State: {state}, Reward: {reward}, Done: {done}")
        env.render()
        pygame.time.delay(500)  # Add a delay of 500 milliseconds (0.5 seconds) between moves

        if done:
            print("Game Over!")
            state = env.reset()  # Reset the game if it's over

    env.close()

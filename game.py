import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize the pygame library
pygame.init()

# Define the font to be used for rendering the score on the screen
font = pygame.font.Font('arial.ttf', 25) # Uses Arial font with size 25
#font = pygame.font.SysFont('arial', 25)


# Define possible directions for the snake as an enumeration for better readability
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


# Define a Point structure with x and y coordinates for use in the game
Point = namedtuple('Point', 'x, y')

# Define RGB color values for use in the game UI
WHITE = (255, 255, 255)  # Color for text
RED = (200, 0, 0)  # Color for the food
BLUE1 = (0, 0, 255)  # Primary color for the snake
BLUE2 = (0, 100, 255)  # Secondary color for the snake for visual effect
BLACK = (0, 0, 0)  # Background color

# Define constants for the game
BLOCK_SIZE = 20  # Size of each snake block and the food
SPEED = 40  # Frames per second (controls game speed)



class SnakeGameAI:
    """ 
    This class encapsulates the entire Snake game logic and rendering.
    It provides methods to initialize the game, update the game state,
    handle user input, and render the game environment. The game is controlled
    programmatically for use with AI agents.
    """
    
    
    def __init__(self, w=640, h=480):
        """
        Initialize the game with specified width and height.
        
        Args:
            w (int): The width of the game window
            h (int): The height of the game window
        
        Attributes:
            w (int): Width of the game window.
            h (int): Height of the game window.
            display (pygame.Surface): The surface object for rendering the game.
            clock (pygame.time.Clock): The clock object to control the game speed.
        """
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))  # Initialize the display
        pygame.display.set_caption('Snake')  # Set the title of the window
        self.clock = pygame.time.Clock()  # Create a clock object for controlling the frame rate
        self.reset()  # Initialize the game state


    def reset(self):
        """
        Resets the game state to its initial configuration.
        This method is useful for restarting the game or initializing it at the start.
        
        Attributes Initialized:
            direction (Direction): The initial direction of the snake (RIGHT).
            head (Point): The initial position of the snake's head (center of the screen).
            snake (list[Point]): List of points representing the snake's body.
            score (int): The player's score, initially 0.
            food (Point): The position of the food (randomly placed).
            frame_iteration (int): Counter to track the number of frames elapsed.
        """
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)  # Start at the center
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0


    def _place_food(self):
        """
        Places food at a random location on the grid.
        Ensures that the food does not overlap with the snake's body.
        
        Attributes Updated:
            food (Point): The new position of the food.
        """
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        """
        Executes one step of the game based on the action taken by the agent.

        Args:
            action (list[int]): A one-hot encoded list representing the agent's action:
                [1, 0, 0] -> Go straight
                [0, 1, 0] -> Turn right
                [0, 0, 1] -> Turn left
        
        Returns:
            tuple: (reward, game_over, score) where:
                reward (int): The reward earned this step (+10 for food, -10 for game over).
                game_over (bool): Whether the game ended in this step.
                score (int): The current score of the player.
        """
        
        self.frame_iteration += 1
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head (the snake's position)
        self.snake.insert(0, self.head) # Add the new head position
        
        # 3. check if game over (collisions)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # 4. place new food or just move (Check if the snake ate the food)
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop() # Remove the tail  
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        
        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        """
        Checks if the snake collides with the boundaries or itself.

        Args:
            pt (Point, optional): The point to check for collision. Defaults to the snake's head.
        
        Returns:
            bool: True if a collision occurs, False otherwise.
        """
        
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        """
        Renders the game environment, including the snake, food, and score.
        """
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        """
        Updates the snake's direction and moves its head accordingly.

        Args:
            action (list[int]): A one-hot encoded list representing the agent's action.
        
        Attributes Updated:
            direction (Direction): The new direction of the snake.
            head (Point): The new position of the snake's head.
        """
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]): # Straight
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]): # Right turn
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # Left turn [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
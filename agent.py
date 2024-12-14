import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot


MAX_MEMORY = 100_000 # Maximum size of the memory buffer
BATCH_SIZE = 1000 # Batch size for training
LR = 0.001 # Learning rate for the optimizer




class Agent:
    """
    The Agent class manages the AI for playing the Snake game. 
    It interacts with the game environment, stores experiences in memory, and trains a neural network using Q-learning.
    """
    
    def __init__(self):
        """
        Initializes the agent with memory, model, and trainer.
        """
        self.n_games = 0  # Count of games played
        self.epsilon = 0  # Exploration parameter for random actions
        self.gamma = 0.9  # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Memory buffer to store experiences : popleft()
        self.model = Linear_QNet(11, 256, 3)  # Neural network model with input, hidden, and output layers
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Trainer to optimize the model
        

    def get_state(self, game):
        """
        Constructs the current state of the game as a feature vector.

        Args:
            game (SnakeGameAI): The game instance.

        Returns:
            np.ndarray: A binary array representing the game state.
        """
        
        head = game.snake[0] # Snake's head position
        point_l = Point(head.x - 20, head.y)  # Point to the left of the head
        point_r = Point(head.x + 20, head.y)  # Point to the right of the head
        point_u = Point(head.x, head.y - 20)  # Point above the head
        point_d = Point(head.x, head.y + 20)  # Point below the head
        
        # Current direction of the snake
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        # State vector representing danger, direction, and food location
        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location relative to the head
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)


    def remember(self, state, action, reward, next_state, done):
        """
        Stores a single experience in memory.

        Args:
            state (np.ndarray): Current state.
            action (list): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after taking the action.
            done (bool): Whether the game ended.
        """
        self.memory.append((state, action, reward, next_state, done)) # Automatically removes oldest if memory is full (popleft if MAX_MEMORY is reached)


    def train_long_memory(self):
        """
        Trains the model on a batch of experiences sampled from memory.
        """
        
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # Randomly sample a batch (list of tuples) 
        else:
            mini_sample = self.memory # Use all stored experiences if memory size is small

        # Unpack experiences into separate lists
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)


    def train_short_memory(self, state, action, reward, next_state, done):
        """
        Trains the model on a single experience.
        """
        self.trainer.train_step(state, action, reward, next_state, done)


    def get_action(self, state):
        """
        Chooses an action based on the current state using an epsilon-greedy strategy.

        Args:
            state (np.ndarray): Current state.

        Returns:
            list: A one-hot encoded action vector.
        """
        
        # random moves: tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games # Decay epsilon as more games are played
        self.epsilon = max(1, 80 - self.n_games)
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            # Random action for exploration
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            # Predict best action using the model
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    """
    The main training loop for the Snake AI agent.
    Continuously plays the game, collects data, trains the model, and plots results.
    """
    
    plot_scores = []  # Scores for each game
    plot_mean_scores = []  # Mean scores over time
    plot_losses = []  # List to store loss values
    total_score = 0
    record = 0  # Highest score achieved
    agent = Agent()
    game = SnakeGameAI()
    
    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory (Train the model with the latest experience)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        
        # Track loss
        loss = agent.trainer.train_step(state_old, final_move, reward, state_new, done)

        # Store the experience in memory (remember)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result (Reset the game and train on long-term memory)
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            # Save the model if the current score is a new record
            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            # Update and plot scores
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_losses.append(loss)  # Add the loss to the list
            plot(plot_scores, plot_mean_scores, plot_losses)  # Pass losses to the plot


if __name__ == '__main__':
    train()
import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import os

MAX_MEMORY = 100_000  # Maximum size of the memory buffer
BATCH_SIZE = 1000  # Batch size for training
LR = 0.001  # Learning rate for the optimizer
max_games = 1000  # Set a maximum number of games


class Agent:
    def __init__(self, model_path='model/model.pth'):
        self.n_games = 0  # Count of games played
        self.epsilon = 0  # Exploration parameter for random actions
        self.gamma = 0.9  # Discount factor for future rewards
        self.memory = deque(maxlen=MAX_MEMORY)  # Memory buffer to store experiences
        self.model = Linear_QNet(11, 256, 3)  # Neural network model
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)  # Trainer for the model
        self.model_path = model_path

        # Check if a pre-trained model exists
        if os.path.exists(self.model_path):
            print(f"Loading existing model from {self.model_path}...")
            self.load_model()
        else:
            print("No existing model found. Starting training from scratch.")

    # def load_model(self):
    #     checkpoint = torch.load(self.model_path)
    #     self.model.load_state_dict(checkpoint['model_state_dict'])
    #     self.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     self.n_games = checkpoint.get('n_games', 0)  # Continue from saved game count
    #     print("Model loaded successfully.")
    
    def load_model(self, file_name='model.pth'):
    # Loads the model's state dictionary directly.
        model_folder_path = './model'
        file_name = os.path.join(model_folder_path, file_name)
        if os.path.exists(file_name):
            print(f"Loading existing model from {file_name}...")
            self.model.load_state_dict(torch.load(file_name))  # Directly load the state dict
        else:
            print("No pre-trained model found. Starting from scratch.")


    def save_model(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.trainer.optimizer.state_dict(),
            'n_games': self.n_games,  # Save the game count
        }, self.model_path)
        print(f"Model saved to {self.model_path}.")

    def get_state(self, game):
        # Same as before
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),

            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),

            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),

            dir_l,
            dir_r,
            dir_u,
            dir_d,

            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y,
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(start_from_scratch=False):
    plot_scores = []
    plot_mean_scores = []
    plot_losses = []
    total_score = 0
    record = 0
    agent = Agent()

    if start_from_scratch:
        print("Starting training from scratch...")
        agent.n_games = 0  # Reset game count
    else:
        print("Using existing model if available...")

    game = SnakeGameAI()

    # while True:
    while agent.n_games < max_games:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        loss = agent.trainer.train_step(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.save_model()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot_losses.append(loss)
            plot(plot_scores, plot_mean_scores, plot_losses)


if __name__ == '__main__':
    # Ask the user whether to start from scratch or continue training
    user_choice = input("Do you want to start training from scratch? (yes/no): ").strip().lower()
    start_from_scratch = user_choice in ['yes', 'y']
    train(start_from_scratch=start_from_scratch)

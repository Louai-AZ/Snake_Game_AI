# Snake Game AI

This project implements an AI agent to play the classic Snake game using Reinforcement Learning. The agent learns to navigate the game environment, avoid collisions, and maximize its score over time by training a neural network using Q-learning.

---

## Features
- **Deep Q-Learning**: The agent uses a neural network to approximate the Q-function.
- **Customizable Training**: The AI agent improves its performance by balancing exploration and exploitation.
- **Real-time Visualization**: Scores and performance are plotted dynamically during training.
- **Scalable Architecture**: Modular design with separate files for game logic, model definition, helper functions, and agent behavior.

---

## Project Structure

- **`game.py`**: Contains the logic for the Snake game, including movement, collision detection, and rendering.
- **`model.py`**: Defines the neural network (Linear_QNet) and Q-learning trainer (QTrainer).
- **`agent.py`**: Implements the AI agent, including state representation, action selection, and training logic.
- **`helper.py`**: Provides utility functions for plotting scores during training.

---

## Getting Started

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/Louai-AZ/Snake_Game_AI.git
    cd Snake_Game_AI
    ```

2. Create an environment and install the dependencies:
    ```bash
        - conda create -n snake_env python=3.7
        - conda activate snake_env
        - pip install pygame
        - pip install torch torchvision
        - pip install matplotlib ipython
    
    ```

3. Run the project:
    ```bash
    python agent.py
    ```

---

## How It Works

### AI Agent
The AI uses a Deep Q-Learning approach:
- The **state** includes information about the snake's surroundings, direction, and the food's location.
- The **action space** has three possible actions: [Straight, Right, Left].
- The **reward system** is designed to encourage the snake to:
  - Move towards the food (+10)
  - Avoid collisions (-10)
  - Stay alive (0 for neutral moves)

### Neural Network
- **Input Layer**: 11 features (state representation)
- **Hidden Layer**: 256 neurons
- **Output Layer**: 3 neurons (one for each action)

### Training
The agent:
1. Explores the environment (using a decaying epsilon for randomness).
2. Stores experiences (state, action, reward, next_state, done) in memory.
3. Trains the neural network using:
   - **Short-term memory**: Immediate experiences.
   - **Long-term memory**: Random batches from past experiences.

---

## Results
- The AI progressively learns to survive longer and achieve higher scores.
- Training progress is visualized in real-time with dynamic plots.

---


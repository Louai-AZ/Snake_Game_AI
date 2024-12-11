import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os



# Neural network class for the Q-Learning model
class Linear_QNet(nn.Module):
    """
    A simple feedforward neural network for Q-learning.

    Args:
        input_size (int): The size of the input layer (number of features in the state).
        hidden_size (int): The size of the hidden layer.
        output_size (int): The size of the output layer (number of possible actions).

    Architecture:
        - Input Layer: Takes the state representation as input.
        - Hidden Layer: A fully connected layer with ReLU activation.
        - Output Layer: Produces Q-values for each possible action.
    """
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Define the first linear layer (input to hidden)
        self.linear1 = nn.Linear(input_size, hidden_size)
        # Define the second linear layer (hidden to output)
        self.linear2 = nn.Linear(hidden_size, output_size)
        

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor with Q-values for each action.
        """
        # Apply ReLU activation to the output of the first layer
        x = F.relu(self.linear1(x))
        # Pass through the second layer to get action Q-values
        x = self.linear2(x)
        return x


    def save(self, file_name='model.pth'):
        # Saves the model's parameters to a file.
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
        # torch.save({'model_state_dict': self.state_dict()}, file_name)


# Trainer class for training the Q-learning model
class QTrainer:
    """
    A trainer class to handle the training process for the Q-learning model.

    Args:
        model (Linear_QNet): The neural network model to train.
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.

    Functionality:
        - Handles forward and backward passes during training.
        - Implements the Q-learning formula to calculate target Q-values.
        - Updates the model's parameters using the loss function.
    """
    
    
    def __init__(self, model, lr, gamma):
        # Learning rate for the optimizer
        self.lr = lr
        # Discount factor for future rewards
        self.gamma = gamma
        # The Q-network model
        self.model = model
        # Adam optimizer for updating the model's weights
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # Mean Squared Error loss for measuring the difference between predicted and target Q-values
        self.criterion = nn.MSELoss()
        self.losses = []  # To store the loss values


    def train_step(self, state, action, reward, next_state, done):
        """
        Performs one step of training using a batch of experience.

        Args:
            state (list or np.ndarray): Current state(s).
            action (list or np.ndarray): Action(s) taken.
            reward (list or np.ndarray): Reward(s) received.
            next_state (list or np.ndarray): Next state(s) after the action.
            done (list or bool): Whether the episode(s) ended.

        Process:
            - Converts inputs to tensors.
            - Calculates predicted Q-values for the current state.
            - Computes target Q-values using the Q-learning formula.
            - Calculates the loss and updates the model parameters.
        """
        
        # Convert inputs to PyTorch tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        # Handle single-dimensional input by adding an extra batch dimension
        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # Step 1: Predict Q-values for the current state
        pred = self.model(state)

        # Clone the predictions to calculate targets
        target = pred.clone()
        
        # Step 2: Compute target Q-values using the Bellman equation
        for idx in range(len(done)):
            Q_new = reward[idx] # If done, target Q-value is just the reward
            if not done[idx]: # If not done, add discounted future reward
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            # Update the target for the action taken
            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        # Step 3: Compute the loss between predicted and target Q-values
        self.optimizer.zero_grad()  # Zero the gradients
        loss = self.criterion(target, pred)  # Calculate loss
        loss.backward()  # Backpropagate the loss

        # Step 4: Update the model parameters
        self.optimizer.step()
        
        self.losses.append(loss.item()) # Store the loss value for plotting
        return loss.item()  # Return the loss for tracking
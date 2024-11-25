import matplotlib.pyplot as plt  
from IPython import display 
#---------------
# Enable interactive mode in matplotlib to update plots dynamically
plt.ion()

# Create the figure and subplots once, outside the function
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))  # Adjusted figure size
fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing

def plot(scores, mean_scores, losses):
    """
    Dynamically plots the training progress of a snake game AI by visualizing scores, mean scores, and losses.
    
    Args:
        scores (list[int]): A list of scores obtained from each game played by the AI.
        mean_scores (list[float]): A list of mean scores over time, calculated as a running average of scores.
        losses (list[float]): A list of loss values recorded during training.

    Functionality:
        - Clears the previous plot and updates it with the latest scores, mean scores, and loss.
        - Dynamically displays the updated plot during training to track AI performance in real time.
        - Annotates the most recent score, mean score, and loss on the plot for clarity.
        - Ensures the y-axis starts at zero for consistency.
    """
    # Clear the previous data on the axes, not the entire figure
    ax1.clear()
    ax2.clear()

    # Plotting Scores and Mean Scores on the first subplot
    ax1.set_title('Training...')  # Title for the first plot
    ax1.set_xlabel('Number of Games')  # X-axis: Number of games played
    ax1.set_ylabel('Score')  # Y-axis: Scores achieved
    ax1.plot(scores, label="Scores")  # Plot the scores over games
    ax1.plot(mean_scores, label="Mean Scores")  # Plot the running average of scores
    ax1.set_ylim(bottom=0)  # Ensure the y-axis starts at 0 for better visualization
    ax1.text(len(scores) - 1, scores[-1], str(scores[-1]))  # Last score
    ax1.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))  # Last mean score
    ax1.legend()

    # Plotting Loss on the second subplot
    ax2.set_title('Training... (Loss)')  # Title for the second plot
    ax2.set_xlabel('Number of Games')  # X-axis: Number of games played
    ax2.set_ylabel('Loss')  # Y-axis: Loss values
    ax2.plot(losses, label='Loss', color='red')  # Plot the loss over games
    ax2.legend()

    # Redraw the current figure and refresh
    plt.draw()
    plt.pause(0.1)  # Short pause to refresh the plot

    # Display the updated plot without opening a new window
    display.display(fig)
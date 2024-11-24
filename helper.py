import matplotlib.pyplot as plt  
from IPython import display 

# Enable interactive mode in matplotlib to update plots dynamically
plt.ion()

def plot(scores, mean_scores):
    """
    Dynamically plots the training progress of a snake game AI by visualizing scores and mean scores.

    Args:
        scores (list[int]): A list of scores obtained from each game played by the AI.
        mean_scores (list[float]): A list of mean scores over time, calculated as a running average of scores.

    Functionality:
        - Clears the previous plot and updates it with the latest scores and mean scores.
        - Dynamically displays the updated plot during training to track AI performance in real time.
        - Annotates the most recent score and mean score on the plot for clarity.
        - Ensures the y-axis starts at zero for consistency.

    Visualization:
        - The x-axis represents the number of games played.
        - The y-axis represents the scores and mean scores.
    """
    # Clear the previous plot to update with the latest data
    display.clear_output(wait=True)
    display.display(plt.gcf())  # Get the current figure and display it dynamically
    plt.clf()  # Clear the figure for fresh plotting

    # Set the title and axis labels for the plot
    plt.title('Training...')  # Indicates the training progress
    plt.xlabel('Number of Games')  # X-axis: Number of games played
    plt.ylabel('Score')  # Y-axis: Scores achieved

    # Plot the individual scores and the mean scores
    plt.plot(scores, label="Scores")  # Plot the scores over games
    plt.plot(mean_scores, label="Mean Scores")  # Plot the running average of scores

    # Ensure the y-axis starts at 0 for better visualization
    plt.ylim(ymin=0)

    # Annotate the most recent values for both scores and mean scores
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))  # Last score
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))  # Last mean score

    # Display the updated plot
    plt.show(block=False)  # Non-blocking display to allow dynamic updates
    plt.pause(0.1)  # Short pause to refresh the plot

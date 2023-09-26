################################
# Plots                        #
# Author: Group 65             #
#                              #
################################

# # Import framwork and other libs
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create a list with all of the .csv files you want to plot
results_list = ['*EA1_enemy2\*EA1_enemy2.csv']

for result in results_list:

    df = pd.read_csv(result)

    # Grouping the DataFrame by 'Gen' (generation) and aggregating the mean and std
    grouped = df.groupby('Gen').agg({'Mean': ['mean', 'std'],
                                    'Best fit': ['mean', 'std']})

    # Flatten the MultiIndex columns
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]

    # Reset index for seaborn plot
    grouped = grouped.reset_index()

    # Set style for seaborn plot
    sns.set_style("whitegrid")

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot mean fitness with std deviation as a shaded region
    sns.lineplot(data=grouped, x='Gen', y='Mean_mean', label='Mean Fitness', color='blue')
    plt.fill_between(grouped['Gen'], 
                    grouped['Mean_mean'] - grouped['Mean_std'], 
                    grouped['Mean_mean'] + grouped['Mean_std'], 
                    color='blue', alpha=0.2)

    # Plot max fitness with std deviation as a shaded region
    sns.lineplot(data=grouped, x='Gen', y='Best fit_mean', label='Best Fit', color='red')
    plt.fill_between(grouped['Gen'], 
                    grouped['Best fit_mean'] - grouped['Best fit_std'], 
                    grouped['Best fit_mean'] + grouped['Best fit_std'], 
                    color='red', alpha=0.2)

    # Set labels and title
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')

    # Add legend
    plt.legend()

    # Get the directory of the currently running Python file
    current_dir = os.path.dirname(os.path.realpath(__file__))
    save_dir = current_dir + '/fitness_plots'

    # Create filename
    name = result.replace('.csv', '')
    file_name = 'fitness_plot_' + name + '.png'

    # Save the plot
    plt.savefig(os.path.join(save_dir, file_name))

    # Show the plot
    plt.show()
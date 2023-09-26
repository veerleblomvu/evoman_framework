################################
# Plots                        #
# Author: Group 65             #
#                              #
################################

# Import framwork and other libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv(f'*EA1_enemy2\*EA1_enemy2.csv')
# print(df)

# Get column names
# for col in df.columns:
#     print(col)


# Assuming your DataFrame is named df
# Grouping the DataFrame by 'generation'
grouped = df.groupby('Gen').agg({'Mean': ['mean', 'std'], 'Best fit': ['mean', 'std']})

df

# Flatten the MultiIndex columns
grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]


# Plotting
plt.figure(figsize=(10, 6))

# Plot mean fitness with std deviation as a shaded region
plt.plot(grouped.index, grouped['Mean_mean'], label='Mean Fitness', color='blue')
plt.fill_between(grouped.index, 
                 grouped['Mean_mean'] - grouped['Mean_std'], 
                 grouped['Mean_mean'] + grouped['Mean_std'], 
                 color='blue', alpha=0.2)

# Plot max fitness with std deviation as a shaded region
plt.plot(grouped.index, grouped['Best fit_mean'], label='Max Fitness', color='red')
plt.fill_between(grouped.index, 
                 grouped['Best fit_mean'] - grouped['Best fit_std'], 
                 grouped['Best fit_mean'] + grouped['Best fit_std'], 
                 color='red', alpha=0.2)

# Set labels and title
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Over Generations')

# Add legend
plt.legend()

# Show the plot
plt.show()
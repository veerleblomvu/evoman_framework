################################
# Plots                        #
# Author: Group 65             #
#                              #
################################

# Import framwork and other libs
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# CREATE LINE PLOTS 
n_generations = 30
enemynr = 3 
enemy = f'enemy{enemynr}'
dataEA1 = pd.read_csv(f'EA1_{enemy}/EA1_{enemy}.csv')
dataEA2 = pd.read_csv(f'EA2_{enemy}/EA2_{enemy}.csv')

#################Enemy _ + EA2########################
####### STD = line ###########
experiment_nameEA2 = f'EA2_enemy{enemynr}'
x_axis = range(n_generations)
group_key = dataEA2.index % 30  # Groups the 1st, 11th, 21st row, and so on, as well as every 2nd row, 12th row, 22nd row, and so on
# Group the DataFrame by the custom key and calculate the means
avg_max_a_EA2 = dataEA2['Best fit'].groupby(group_key).mean()
std_max_EA2 = dataEA2['Best fit'].groupby(group_key).std()
avg_avg_a_EA2 = dataEA2['Mean'].groupby(group_key).mean()
std_avg_EA2 = dataEA2['Mean'].groupby(group_key).std()    
plt.figure("EA2 - {enemynr} Line plot fitness")
plt.plot(x_axis,avg_max_a_EA2)
plt.plot(x_axis, avg_avg_a_EA2)
plt.plot(x_axis, std_avg_EA2)
plt.plot(x_axis, std_max_EA2)
plt.title("EA2 - {enemynr} Line plot fitness")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(['avg max', 'avg mean', 'std mean', 'std max'])
plt.savefig(f'{experiment_nameEA2}\{experiment_nameEA2} lineplot2.png')  # 2 (i want to check) because in the original file we also save one (should be the same)
plt.show()

####### STD = spread ###########
plt.figure("EA2 - {enemynr} Line plot fitness + spread")
plt.plot(x_axis,avg_avg_a_EA2)
plt.fill_between(x_axis, avg_avg_a_EA2-std_avg_EA2, avg_avg_a_EA2+std_avg_EA2, alpha=0.5)
plt.plot(x_axis, avg_max_a_EA2)
plt.fill_between(x_axis, avg_max_a_EA2-std_max_EA2, avg_max_a_EA2+std_max_EA2, alpha=0.5)
plt.title("EA2 - {enemynr} Line plot fitness + spread")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(['avg mean', 'avg max', 'std mean', 'std max'])
plt.savefig(f'{experiment_nameEA2}\{experiment_nameEA2} lineplotSpread.png')  # 2 (i want to check) because in the original file we also save one (should be the same)
plt.show()


#################Enemy _ + EA1 ########################
####### STD = line ###########
experiment_nameEA1 = f'EA1_enemy{enemynr}'
x_axis = range(n_generations)
group_key = dataEA1.index % 30  # Groups the 1st, 11th, 21st row, and so on, as well as every 2nd row, 12th row, 22nd row, and so on
# Group the DataFrame by the custom key and calculate the means
avg_max_a_EA1 = dataEA1['Best fit'].groupby(group_key).mean()
std_max_EA1 = dataEA1['Best fit'].groupby(group_key).std()
avg_avg_a_EA1 = dataEA1['Mean'].groupby(group_key).mean()
std_avg_EA1 = dataEA1['Mean'].groupby(group_key).std()    
plt.figure(f"EA1 - {enemynr} Line plot fitness")
plt.plot(x_axis,avg_max_a_EA1)
plt.plot(x_axis, avg_avg_a_EA1)
plt.plot(x_axis, std_avg_EA1)
plt.plot(x_axis, std_max_EA1)
plt.title(f"EA1 - {enemynr} Line plot fitness")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(['avg max', 'avg mean', 'std mean', 'std max'])
plt.savefig(f'{experiment_nameEA1}\{experiment_nameEA1} lineplot2.png')  # 2 (i want to check) because in the original file we also save one (should be the same)
plt.show()
   
plt.figure(f"EA1 - {enemynr} Line plot fitness + Spread")
plt.plot(x_axis,avg_avg_a_EA1)
plt.fill_between(x_axis, avg_avg_a_EA1-std_avg_EA1, avg_avg_a_EA1+std_avg_EA1, alpha=0.5)
plt.plot(x_axis, avg_max_a_EA1)
plt.fill_between(x_axis, avg_max_a_EA1-std_max_EA1, avg_max_a_EA1+std_max_EA1, alpha=0.5)
plt.title(f"EA1 - {enemynr} Line plot fitness + Spread")
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(['avg mean', 'avg max', 'std mean', 'std max'])
plt.savefig(f'{experiment_nameEA1}\{experiment_nameEA1} lineplotSpread.png')  # 2 (i want to check) because in the original file we also save one (should be the same)
plt.show()



################################
# EvoMan FrameWork - EA2       #
# Author: Group 65             #
#                              #
################################

# Import framwork and other libs
import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
import os
import random
import pandas as pd
import matplotlib.pyplot as plt
import csv


# Choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

enemy = 8 # MAKE SURE TO ALSO CHANGE LINE 36
# Create a folder for the experiment in which all the data are stored
experiment_name = f'EA2_enemy{enemy}'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# The NN has one hidden layer with 10 neurons
n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

# OPTIMIZATION FOR CONTROLLER SOLUTION: GENETIC ALGORITHM

# Genetic algorithm parameters
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 # number of weights for multilayer with 10 hidden neurons
upper_bound = 1 # upper bound of start weights
lower_bound = -1 # lower bound of start weights
pop_size = 100
n_generations = 30
n_runs = 10
mutation_rate = 0.2
sigma = 1
learning_rate = 0.1
boundary = 0.001
k = 2


def initialize(population_size, lower_bound, upper_bound, n_weights):
    return np.random.uniform(lower_bound, upper_bound, (population_size, n_weights))


def simulation(env,x):
    """
    Run a simulation for one individual in the population.

    Parameters:
        env (Environment): The environment object representing the game.
        pop (array-like): The controller or policy for the individual.

    Returns:
        float: The fitness score and gain of the individual.
    """
    # Run the simulation and get fitness, player energy, enemy energy, and duration
    fit,p_energy,e_energy,duration = env.play(pcont=x)
    gain = p_energy - e_energy
    return fit, gain

def evaluate(pop):
    """
    Determine the fitnesses of individuals in the population.

    Parameters:
        pop (list): The population of individuals.

    Returns:
        numpy.ndarray: An array containing the fitness score for each individual.
    """
    pop_fit_gain = np.array([simulation(env, y) for y in pop])
    return pop_fit_gain

def tournament(pop, pop_fit, k):
    """
    Perform a tournament selection on a population.

    Parameters:
        pop (numpy.ndarray): The population of individuals.
        pop_fit (numpy.ndarray): The fitness scores of the individuals.
        k (int): The number of individuals competing in each tournament.

    Returns:
        numpy.ndarray: The winning individual from the tournament.
    """
    n_individuals = pop.shape[0]
    current_winner = np.random.randint(0, n_individuals-1)
    current_max_fit = pop_fit[current_winner]

    for candidates in range(k-1): #We already have one candidate, so we are left with k-1 to choose
        contender_number = np.random.randint(0, n_individuals-1)
        if pop_fit[contender_number] > current_max_fit:
            current_winner = contender_number
            current_max_fit = pop_fit[contender_number]
    winner = pop[current_winner]
    return winner
    
def limits(x, lower_bound, upper_bound):
    """
    Ensure x is within specified bounds.

    Parameters:
        x (float): The input value.
        lower_bound (float): The lower bound.
        upper_bound (float): The upper bound.

    Returns:
        float: The bounded value.
    """
    if x>upper_bound:
        return upper_bound
    elif x<lower_bound:
        return lower_bound
    else:
        return x

def whole_arithmic_crossover(pop, pop_fit, k, alpha=0.5):
    """
    Perform whole arithmetic crossover on a population.

    Parameters:
        pop (numpy.ndarray): The population of individuals.
        pop_fit (numpy.ndarray): The fitness scores of the individuals.
        k (int): The number of individuals competing in each tournament.
        alpha (float, optional): The blending factor. Default is 0.5.

    Returns:
        numpy.ndarray: The resulting offspring population.
    """
    offspring = []
    for p in range(0, pop.shape[0], 2):
        parent1 = tournament(pop, pop_fit, k)
        parent2 = tournament(pop, pop_fit, k)
        child1 = []
        child2 = []
        for gene1, gene2 in zip(parent1, parent2):
            offspring1 = alpha * gene1 + (1 - alpha) * gene2
            offspring2 = alpha * gene2 + (1 - alpha) * gene1
            child1.append(offspring1)
            child2.append(offspring2)
        offspring.extend([child1, child2])
    return np.array(offspring)

def update_sigma(sigma, learning_rate, boundary):
    """
    Update the value of sigma.

    Args:
        sigma (float): The previous value of sigma.
        learning_rate (float): The learning rate.
        boundary (float): The boundary value.

    Returns:
        float: The updated value of sigma.
    """
    exponent = np.exp(learning_rate * (np.random.normal(0,1)))
    sigma = sigma * exponent
    if sigma < boundary:
        sigma = boundary
    return sigma

def self_adapt_mutate(offspring, mutation_rate, sigma):
    """
    Apply self adaptive mutation with one step size to the offspring.

    Args:
        offspring (numpy.ndarray): The offspring population.
        mutation_rate (float): The mutation rate.
        sigma (float): The sigma value.

    Returns:
        numpy.ndarray: The mutated offspring population.
    """
    for i in range(len(offspring)):
        if np.random.uniform(0,1) <= mutation_rate:
            offspring[i] += sigma * np.random.normal(0,1)
    offspring = np.array([limits(y, -1, 1) for y in offspring])
    return offspring

def elitism(pop, pop_fit, x):
    """
    Select the best x individuals from the population based on fitness.

    Parameters:
        pop (list): The list of individuals.
        pop_fit (list): The fitness scores of the individuals.
        pop_gain (list): The gain of the individuals.
        x (int): The number of best individuals to select.

    Returns:
        tuple: A tuple containing 2 lists - best individuals, their fitness scores
    """
    # Sort the population by fitness (in descending order) and get the indices of the best individuals
    sorted_fit_indices = np.argsort(pop_fit)[::-1]
    best_indices = sorted_fit_indices[:x]
    
    # Select the best individuals and their fitness scores and gains
    best_pop = [pop[i] for i in best_indices]
    best_pop_fit = [pop_fit[i] for i in best_indices]

    return best_pop, best_pop_fit

def elitism_survival_selection(parents, parents_fit, offspring, x, y):
    """
    Perform survival selection to create the next generation.

    This function combines the fittest parents and randomly selected children to form the new population.

    Parameters:
        parents (numpy.ndarray): The parent individuals.
        parents_fit (numpy.ndarray): The fitness scores of the parents.
        offspring (numpy.ndarray): The child individuals.
        x (int): The number of fittest parents to keep.
        y (int): The number of random children to keep.

    Returns:
        numpy.ndarray: The new population.
    """
    # Check if x and y are correct values
    if (x + y) != 100 or x < 0 or y < 0 or x > 100 or y > 100:
        raise ValueError("The values of x and y are incorrect.")
    
    # Select the x fittest parents
    best_parents, best_parents_fit = elitism(parents, parents_fit, x)

    # Select y children from offspring
    random_indices = random.sample(range(len(offspring)), y)
    selected_offspring = [offspring[i] for i in random_indices]

    # Combine the best parents and random children to form the new population
    pop = np.vstack((best_parents, selected_offspring))

    return pop



indices=[]
best_gain = []
best_fit = []
mean_fitness = []
std_fitness = []
gen = []
best_solutions = []
result_matrix_max=np.zeros((n_runs,n_generations))
result_matrix_mean=np.zeros((n_runs,n_generations))

# EVOLUTIONARY LOOP
for r in range(n_runs):
    i = 0
    pop = initialize(pop_size, lower_bound, upper_bound, n_weights)
    pop_fit_gain = evaluate(pop)
    pop_fit = pop_fit_gain[:,0]
    pop_gain = pop_fit_gain[:,1]
    best = np.argmax(pop_fit)
    best_solution = pop[best]
    best_solutions.append(best_solution)

    mean = np.mean(pop_fit)
    std = np.std(pop_fit)
    result_matrix_max[r,i]=np.max(pop_fit)
    result_matrix_mean[r,i]=np.mean(pop_fit)
    indices.append([f"Test {r} - Gen {i}"])
    best_gain.append(pop_gain[best])
    best_fit.append(pop_fit[best])
    mean_fitness.append(mean)
    std_fitness.append(std)
    gen.append(i)
    # Saves result
    experiment_data  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    experiment_data.write('\n'+str(r)+' '+str(i)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    experiment_data.close()

# Loop through generations
    for i in range(1,n_generations):
        # Create offspring applying crossover and mutation
        offspring = whole_arithmic_crossover(pop, pop_fit, k, alpha=0.5)
        offspring = [self_adapt_mutate(gene, mutation_rate, sigma) for gene in offspring]
        sigma = update_sigma(sigma, learning_rate, boundary)
        
        # Survival selection (10 elite parents + 90 random children)
        pop = elitism_survival_selection(pop, pop_fit, offspring, 10, 90)
        pop_fit_gain = evaluate(pop)
        pop_fit = pop_fit_gain[:,0]
        pop_gain = pop_fit_gain[:,1]

        best = np.argmax(pop_fit)
        best_solution = pop[best]
        best_solutions.append(best_solution)
        std  =  np.std(pop_fit)
        mean = np.mean(pop_fit)

        result_matrix_max[r,i]=np.max(pop_fit)
        result_matrix_mean[r,i]=np.mean(pop_fit)
        indices.append([f"Test {r} - Gen {i}"])
        best_gain.append(pop_gain[best])
        best_fit.append(pop_fit[best])
        mean_fitness.append(mean)
        std_fitness.append(std)
        gen.append(i)

        # Saves result
        experiment_data  = open(experiment_name+'/results.txt','a')
        print( '\n GENERATION '+str(i)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
        experiment_data.write('\n'+str(r)+' '+str(i)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
        experiment_data.close()


print("len ind", len(indices))
print("len gain", len(best_gain))
print("len fit", len(best_fit))
print("len mean", len(mean_fitness))
print("len std", len(std_fitness))

d = {"Run": indices, "gain": best_gain, "Best fit": best_fit, "Mean": mean_fitness, "STD": std_fitness,"BEST SOL":best_solutions}
df = pd.DataFrame(data=d)
print(df)
# makes csv file
df.to_csv(f'{experiment_name}\{experiment_name}.csv', index = False)

plt.figure("Boxplot")
plt.boxplot(best_gain)
plt.savefig(f'{experiment_name}\{experiment_name} boxplot_gain.png')
plt.show()

x_axis = range(n_generations)
avg_avg_a = np.mean(result_matrix_mean,axis=0)
avg_max_a = np.mean(result_matrix_max,axis=0)
std_avg = np.std(result_matrix_mean,axis=0)
std_max = np.std(result_matrix_max,axis=0)
plt.figure("Line plot")
plt.plot(x_axis,avg_avg_a)
plt.plot(x_axis, avg_max_a)
plt.plot(x_axis, std_avg)
plt.plot(x_axis, std_max)
plt.title('Statistics across generations')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.legend(['avg mean', 'avg max', 'std mean', 'std max'])
plt.savefig(f'{experiment_name}\{experiment_name} lineplot.png')
plt.show()
#prints the matrices
# print(np.mean(result_matrix_mean,axis=0))
# print(np.mean(result_matrix_max,axis=0))
# print(np.std(result_matrix_mean,axis=0))
# print(np.std(result_matrix_max,axis=0))

print(result_matrix_max)
#env.state_to_log() # checks environment state
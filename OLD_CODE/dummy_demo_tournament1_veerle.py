################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Group 65             #
#                              #
################################

# Import framework
from random import randint
import sys, os
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import pandas as pd
import matplotlib.pyplot as plt
import random


experiment_name = 'group_65_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# Choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# The network has one layer of 10 hidden neurons
n_hidden_neurons = 10

# Initialize simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# Default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
ini = time.time()  # sets time marker

# Genetic algorithm params
run_mode = 'train' # train or test

def simulation(env,pop):
    """
    Run a simulation for one individual in the population.

    Parameters:
        env (Environment): The environment object representing the game.
        pop (array-like): The controller or policy for the individual.

    Returns:
        tuple: A tuple containing fitness, player energy, enemy energy, duration, and gain.
            - fitness (float): The fitness score of the individual.
            - p_energy (float): The energy of the player.
            - e_energy (float): The energy of the enemy.
            - duration (float): The duration of the game.
            - gain (float): The energy gain (player energy - enemy energy).
    """
    # Run the simulation and get fitness, player energy, enemy energy, and duration
    fitness,p_energy,e_energy,duration = env.play(pcont=pop)
    
    # Calculate energy gain (player energy - enemy energy)
    gain = p_energy - e_energy
    
    # Check if the player won (has positive energy)
    if p_energy > 0:
        print("Gewonnen!")
    
    return fitness, p_energy, e_energy, duration, gain

def evaluate(pop):
    """
    Evaluate the fitness of individuals in the population.

    Parameters:
        pop (numpy.ndarray): The population of individuals.

    Returns:
        numpy.ndarray: An array containing fitness scores and gains for each individual.
    """
    # Apply the simulation function to each individual in the population
    fitness_and_gains = np.array(list(map(lambda y: np.array(simulation(env,y))[[0,4]], pop)))
    return fitness_and_gains

def initialize(population_size, lower_bound, upper_bound, n_weights):
    """
    Initialize the first generation of the experiment.

    Parameters:
        population_size (int): The size of the population.
        lower_bound (float): The lower bound for weight initialization.
        upper_bound (float): The upper bound for weight initialization.
        n_weights (int): The number of weights in an individual.

    Returns:
        numpy.ndarray: A population with randomly initialized weights.
    """
    # Generate a population with random weights within the specified bounds
    population = np.random.uniform(lower_bound, upper_bound, (population_size, n_weights))
    return population

def parent_selection(pop, pop_fit, n_parents, smoothing = 1):
    """
    Select parents from the population based on fitness.

    Parameters:
        pop (numpy.ndarray): The population of individuals.
        pop_fit (numpy.ndarray): The fitness scores of the population.
        n_parents (int): The number of parents to select.
        smoothing (float, optional): A smoothing factor to ensure all individuals have a chance of being selected.
                                     Default is 1.

    Returns:
        numpy.ndarray: The selected parents.
    """
    fitness = pop_fit + smoothing - np.min(pop_fit)
    # Fitness proportional selection probability
    fps = fitness / np.sum(fitness)
    # Make a random selection of indices
    parent_indeces = np.random.choice(np.arange(0, pop.shape[0]), (n_parents, 2), p=fps)
    return pop[parent_indeces]

def tournament_selection(pop: list, pop_fit: list, k: int) -> list:
    """
    Perform tournament selection to choose a parent.

    Parameters:
        pop (numpy.ndarray): The population of individuals.
        pop_fit (numpy.ndarray): The fitness scores of the population.
        k (int): The number of individuals in each tournament.

    Returns:
        int: The index of the selected parent.
    """
    number_individuals = pop.shape[0]
    current_winner = randint(0, number_individuals-1)
    score = pop_fit[current_winner]
    
    for candidates in range(k-1):
        contender_number = randint(0, number_individuals-1)
        if pop_fit[contender_number] > score:
            current_winner = contender_number
            score = pop_fit[contender_number]
            
    return current_winner

def uniform_crossover(parents): 
    """
    Create a child from two parents using uniform crossover.

    Uniform crossover means each gene has an equal chance of coming from either parent.

    Parameters:
        parents (list of numpy.ndarray): The two parents for crossover.

    Returns:
        numpy.ndarray: The child resulting from uniform crossover.
    """
    parent1 = parents[0]
    parent2 = parents[1]
    # Generate an array of random numbers between 0 and 1 to determine which genes come from which parent
    random_list = np.random.uniform(size=parent1.shape)
    
    # Generate the child by selecting genes from the parents based on the random numbers
    child = parent1 * (random_list>=0.5) + parent2 * (random_list < 0.5)
    
    # Squeeze to remove any extra dimensions that may have been created during parent selection    
    return np.squeeze(child)

def uniform_mutate(child, mutation_rate):
    """
    Apply uniform mutation to a child.

    Parameters:
        child (numpy.ndarray): The individual to be mutated.
        mutation_rate (float): The probability of a gene being mutated.

    Returns:
        numpy.ndarray: The mutated child.
    """
    # Iterate over weights in the individual
    for i in range(len(child)):
        # Check if a mutation should occur for this weight
        if np.random.uniform(0, 1) <= mutation_rate: 
            # Apply mutation by setting the weight to a random value in the range [0, 1]
            child[i] = np.random.uniform(0, 1)
            
    return child

def elitism(pop: list, pop_fit: list, pop_gain: list, x: int):
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
    best_pop_gain = [pop_gain[i] for i in best_indices]

    return best_pop, best_pop_fit, best_pop_gain

def survival_selection(parents, parents_fit, parents_gain, offspring, offspring_fit, offspring_gain, x):
    """
    Perform survival selection to create the next generation.

    This function combines the fittest parents and randomly selected children to form the new population.

    Parameters:
        parents (numpy.ndarray): The parent individuals.
        parents_fit (numpy.ndarray): The fitness scores of the parents.
        parents_gain (numpy.ndarray): The energy gains of the parents.
        offspring (numpy.ndarray): The child individuals.
        offspring_fit (numpy.ndarray): The fitness scores of the children.
        offspring_gain (numpy.ndarray): The energy gains of the children.
        x (int): The number of fittest parents to keep.

    Returns:
        tuple: A tuple containing the new population, fitness scores, and energy gains.
            - pop (numpy.ndarray): The combined population.
            - pop_fit (numpy.ndarray): The fitness scores of the combined population.
            - pop_gain (numpy.ndarray): The energy gains of the combined population.
    """
    # Select the x fittest parents
    best_parents, best_parents_fit, best_parents_gain = elitism(parents, parents_fit, parents_gain, x)

    # Print some information for debugging or monitoring purposes
    # print(f"Number of best parents selected: {len(best_parents)}")
    # print(f"Fitness scores of best parents: {best_parents_fit}")
    # print(f"Energy gains of best parents: {best_parents_gain}")

    # Select population size - x random children from offspring
    pop_size = len(parents)
    x_remaining = pop_size - x
    random_indices = random.sample(range(len(offspring)), x_remaining)
    selected_offspring = [offspring[i] for i in random_indices]
    selected_offspring_fit = [offspring_fit[i] for i in random_indices]
    selected_offspring_gain = [offspring_gain[i] for i in random_indices]

    # Combine the best parents and random children to form the new population
    pop = np.vstack((best_parents, selected_offspring))
    pop_fit = np.concatenate([best_parents_fit, selected_offspring_fit])
    pop_gain = np.concatenate([best_parents_gain, selected_offspring_gain])

    return pop, pop_fit, pop_gain


# parameters
upper_bound = 1 # upper bound of start weights
lower_bound = -1 # lower bound of start weights
pop_size = 100
generations = 15
mutation_rate = 0.5
last_best = 0
# number of weights for multilayer with 10 hidden neurons
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
n_runs= 10
n_tests=5
best_results=[]

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
pop = initialize(pop_size, lower_bound, upper_bound, n_weights)
pop_fit = evaluate(pop)
result_matrix_max=np.zeros((n_runs,generations))
result_matrix_mean=np.zeros((n_runs,generations))

#lists for pandas dataframe for csv file columns
indices=[]
best_gain = []
best_fit = []
mean = []
std = []
gen = []

# Run every evolution n times
for y in range(n_runs):
    # Initialize the population
    pop = initialize(pop_size, lower_bound, upper_bound, n_weights)
    
    # Evaluate the fitness of the population
    pop_result = evaluate(pop)
    pop_fit=pop_result[:,0]
    pop_gain=pop_result[:,1]
    
    # Generate within the evolution
    for i in range(generations):
        offspring = []
 
        # Create n children out of n pairs of parents using tournament selection
        for j in range(int(pop_size)):
            parent1_index = tournament_selection(pop, pop_fit, 10)
            parent2_index = tournament_selection(pop, pop_fit, 10)
            parent1 = pop[parent1_index]
            parent2 = pop[parent2_index]
            parents = [parent1, parent2]
            
            child = uniform_crossover(parents)
            child = uniform_mutate(child, mutation_rate)
            offspring.append(child)
        
        offspring_result = evaluate(offspring)
        offspring_gain=offspring_result[:,1]
        offspring_fit=offspring_result[:,0]

        # Survival selection (10 elite parents + 90 random children)
        pop, pop_fit, pop_gain = survival_selection(pop, pop_fit, pop_gain, offspring, offspring_fit, offspring_gain, 10)

        best_fitness_index = np.argmax(pop_fit)
        print(f"Test {y} - Gen {i} - Best_fit: {pop_fit[best_fitness_index]} - Best_gain: {pop_gain[best_fitness_index]} - Mean: {np.mean(pop_fit)} - Std: {np.std(pop_fit)}")
        result_matrix_max[y,i]=np.max(pop_fit)
        result_matrix_mean[y,i]=np.mean(pop_fit)
        indices.append([f"Test {y} - Gen {i}"])
        best_gain.append(pop_gain[best_fitness_index])
        best_fit.append(pop_fit[best_fitness_index])
        mean.append(np.mean(pop_fit))
        std.append(np.std(pop_fit))
        gen.append(i)

# makes dataframe to make into csv file
d = {"Run": indices, "gain": best_gain, "Best fit": best_fit, "Mean": mean, "STD": std}
df = pd.DataFrame(data=d)
print(df)
# makes csv file
df.to_csv('pandas_results.csv', index=False)

plt.figure("Boxplot")
plt.boxplot(best_gain)
plt.show()
plt.savefig('boxplot_gain.png')

#line plot
x_axis = range(generations)
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
plt.show()
plt.savefig('lineplot.png')
#prints the matrices
print(np.mean(result_matrix_mean,axis=0))
print(np.mean(result_matrix_max,axis=0))
print(np.std(result_matrix_mean,axis=0))
print(np.std(result_matrix_max,axis=0))
# env.play()


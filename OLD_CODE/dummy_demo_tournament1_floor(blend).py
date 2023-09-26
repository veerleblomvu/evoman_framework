################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Group 65             #
#                              #
################################

# imports framework
# test change - 12-09-2023
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

experiment_name = 'group_65_test'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

n_hidden_neurons = 10

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[5],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)

# default environment fitness is assumed for experiment
env.state_to_log() # checks environment state

####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###
ini = time.time()  # sets time marker

# genetic algorithm params
run_mode = 'train' # train or test

# runs simulation (one game by one player)
def simulation(env,pop):
    fitness,p_energy,e_energy,duration = env.play(pcont=pop)
    gain = p_energy - e_energy
    return fitness, p_energy, e_energy, duration, gain

# adds fitness for all individuals in population to a list
def evaluate(pop):
    return np.array(list(map(lambda y: np.array(simulation(env,y))[[0,4]], pop)))

# initializes the first generation of our experiment
def initialize(population_size, lower_bound, upper_bound, n_weights):
    return np.random.uniform(lower_bound, upper_bound, (population_size, n_weights))


# K random individuals (with replacement) are chosen and compete with each other. The index of the best individual is returned.
def tournament_selection(pop: list, pop_fit: list, k: int) -> list:
    #First step: Choose a random individual and score it
    number_individuals = pop.shape[0]
    current_winner = randint(0, number_individuals-1)
    #Get the score which is the one to beat!
    score = pop_fit[current_winner]
    
    for candidates in range(k-1): #We already have one candidate, so we are left with k-1 to choose
        contender_number = randint(0, number_individuals-1)
        if pop_fit[contender_number] > score:
            current_winner = contender_number
            score = pop_fit[contender_number]
            
    return current_winner


# creates one child out of two parents, using uniform crossover
def uniform_crossover(parents): 
    parent1 = parents[0]
    parent2 = parents[1]
    random_list = np.random.uniform(size=parent1.shape)
    offspring = parent1 * (random_list>=0.5) + parent2 * (random_list < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return offspring

def blend_crossover(parents, alpha =0.5):
    parent1 = parents[0]
    parent2 = parents[1]
    offspring = []
    child1 = []
    child2=[]
    for gene1, gene2 in zip(parent1, parent2):
        random = np.random.uniform(0,1)
        gamma = (1 - 2*alpha)*random - alpha
        offspring1 = (1-gamma)*gene1 + gamma*gene2
        child1.append(offspring1)
        offspring2 = (1-gamma)*gene2 + gamma*gene1  
        child2.append(offspring2)
        offspring = child1, child2  #creates a tuple where offspring[0] = child1 and offspring[1]= child2
    return offspring

def uniform_mutate(offspring, mutation_rate):
    for i in range(0, n_weights): 
        if np.random.uniform(0,1) <= mutation_rate: 
            offspring[i] = np.random.uniform(-1, 1)
    return offspring

# returns best half of the population and their fitnesses
def survivor_selection(population, population_fitness, population_size):
    best_fit_indices = np.argsort(population_fitness * -1) # -1 since we are maximizing
    survivor_indices = best_fit_indices [:population_size]
    return population[survivor_indices], population_fitness[survivor_indices]

# parameters
upper_bound = 1 # upper bound of start weights
lower_bound = -1 # lower bound of start weights
pop_size = 100
generations = 30
mutation_rate = 0.2
last_best = 0
# number of weights for multilayer with 10 hidden neurons
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5
n_runs= 10
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

# evolution
for y in range(n_runs):
    pop = initialize(pop_size, lower_bound, upper_bound, n_weights)
    pop_result = evaluate(pop)
    pop_gain=pop_result[:,1]
    pop_fit=pop_result[:,0]
    for i in range(generations):
        # parents_original = parent_selection(pop, pop_fit, pop_size)
        offspring = []
        for j in range(int(pop_size/2)):
            parent1_index = tournament_selection(pop, pop_fit, k=2)
            parent1 = pop[parent1_index]
            parent2_index = tournament_selection(pop, pop_fit, k=2)
            parent2 = pop[parent2_index]
            parents = [parent1, parent2]
            #offspring_individual = uniform_crossover(parents)
            #offspring_individual = uniform_mutate(offspring_individual, mutation_rate)
            #offspring.append(offspring_individual)
            offspring1= blend_crossover(parents, alpha=0.5)[0]
            offspring1 = uniform_mutate(offspring1, mutation_rate)

            offspring2 = blend_crossover(parents, alpha=0.5)[1]
            offspring1 = uniform_mutate(offspring2, mutation_rate)
            offspring += offspring1 , offspring2
            #print("how many offspring", len(offspring))

        offspring_result = evaluate(offspring)
        offspring_gain=offspring_result[:,1]
        offspring_fit=offspring_result[:,0]
        pop = np.vstack((pop, offspring))
        pop_fit = np.concatenate([pop_fit, offspring_fit])
        pop_gain = np.concatenate([pop_gain, offspring_gain])
        pop, pop_fit= survivor_selection(pop, pop_fit, pop_size)
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


################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Group 65             #
#                              #
################################

# imports framework
# test change - 12-09-2023
import sys, os
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os

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
                  enemies=[1],
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
    return fitness, p_energy, e_energy, duration

# adds fitness for all individuals in population to a list
def evaluate(pop):
    return np.array(list(map(lambda y: simulation(env,y)[0], pop)))

# initializes the first generation of our experiment
def initialize(population_size, lower_bound, upper_bound, n_weights):
    return np.random.uniform(lower_bound, upper_bound, (population_size, n_weights))

# # returns the mean and the maximum population fitness
# def evaluate_pop(pop):
#     fitnesses = evaluate(pop)
#     return np.mean(fitnesses), np.max(fitnesses)

# creates n pairs of parents withing a population
# individuals with a higher fitness have a higher chance of becoming parent
# individuals with worst fitness, still have chance of being selected due to smoothing
def parent_selection(pop, pop_fit, n_parents, smoothing = 1):
    fitness = pop_fit + smoothing - np.min(pop_fit)
    # Fitness proportional selection probability
    fps = fitness / np.sum(fitness)
    # make a random selection of indices
    parent_indeces = np.random.choice(np.arange(0, pop.shape[0]), (n_parents, 2), p=fps)
    return pop[parent_indeces]

# creates one child out of two parents, using uniform crossover
def uniform_crossover(parents): 
    parent1, parent2 = np.hsplit(parents,2)
    section = np.random.uniform(size=parent1.shape)
    offspring = parent1 * (section>=0.5) + parent2 * (section < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return np.squeeze(offspring, 1)

def whole_arithmic_crossover(parents, alpha=0.5):
    parent1 = parents[0]
    parent2 = parents[1]
    offspring = []
    child1 = []
    child2=[]
    for gene1, gene2 in zip(parent1, parent2):
        offspring1 = alpha * gene1 + (1-alpha) * gene2
        child1.append(offspring1)
        offspring2 = alpha * gene2 + (1-alpha) * gene1
        child2.append(offspring2)
        offspring = child1, child2  #creates a tuple where offspring[0] = child1 and offspring[1]= child2
    return offspring
def update_sigma(sigma, learning_rate):
    '''Takes the previous sigma and updates it
    Args: sigma (float)
    learning_rate (float) set value by user'''
    random_value = np.random.normal(0,1)
    exponent = np.exp(learning_rate * random_value)
    sigma = sigma * exponent
    if sigma < boundary:
        sigma = boundary
    return sigma

# mutates the offspring using uniform mutation
def uniform_mutate(offspring, mutation_rate, sigma):
    '''takes the offspring and mutates it
    
    Args: offspring (list)'''
    # iterate over individuals in offspring
    for i in range(len(offspring)):
        # iterate over weights in individual
        for j in range(len(offspring[i])): 
            # with probability mutation_rate, alter weight in individual
            # with value sampled from uniform distribution.
            if np.random.uniform(0,1) <= mutation_rate: 
                offspring[i][j] += sigma * np.random.normal(0, 1)
        np.array(list(map(limits, offspring[i])))      
    return offspring  

def limits(value):
    return max(min(value, upper_bound), lower_bound)

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
learning_rate = 0.1
sigma = 1.0
boundary = 0.2
# number of weights for multilayer with 10 hidden neurons
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
pop = initialize(pop_size, lower_bound, upper_bound, n_weights)
pop_fit = evaluate(pop)

# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    # pop = np.random.uniform(lower_bound, upper_bound, (pop_size, n_weights))
    fit_pop = evaluate(pop)
    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)
    ini_g = 0
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]

    best = np.argmax(fit_pop)
    mean = np.mean(fit_pop)
    std = np.std(fit_pop)

    # finds last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    ini_g = int(file_aux.readline())
    file_aux.close()

# saves results for first pop
# essentially initializes the save file
# from optimization_specialist_demo.py
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()

# evolution
for i in range(generations):
    parents = parent_selection(pop, pop_fit, pop_size)
    offspring = uniform_crossover(parents)
    sigma = update_sigma(sigma, learning_rate)
    print(sigma)
    offspring = uniform_mutate(offspring, mutation_rate, sigma)
    offspring_fit = evaluate(offspring)
    pop = np.vstack((pop, offspring))
    pop_fit = np.concatenate([pop_fit, offspring_fit])
    pop, pop_fit = survivor_selection(pop, pop_fit, pop_size)
    print (f"Gen {i} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")

    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.write(f"Gen {i} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")
    file_aux.close()

    # saves generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # saves simulation state
    solutions = [pop, fit_pop]
    env.update_solutions(solutions)
    env.save_state()
# env.play()


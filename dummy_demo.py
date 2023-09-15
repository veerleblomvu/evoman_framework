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

# mutates the offspring using uniform mutation
def uniform_mutate(offspring, mutation_rate):
    # iterate over individuals in offspring
    for i in range(len(offspring)):
        # iterate over weights in individual
        for j in range(len(offspring[i])): 
            # with probability mutation_rate, alter weight in individual
            # with value sampled from uniform distribution.
            if np.random.uniform(0,1) <= mutation_rate: 
                offspring[i][j] = np.random.uniform(0, 1)
            
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

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name)
pop = initialize(pop_size, lower_bound, upper_bound, n_weights)
pop_fit = evaluate(pop)

# evolution
for i in range(generations):
    parents = parent_selection(pop, pop_fit, pop_size)
    offspring = uniform_crossover(parents)
    offspring = uniform_mutate(offspring, mutation_rate)
    offspring_fit = evaluate(offspring)
    pop = np.vstack((pop, offspring))
    pop_fit = np.concatenate([pop_fit, offspring_fit])
    pop, pop_fit = survivor_selection(pop, pop_fit, pop_size)
    print (f"Gen {i} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")
# env.play()


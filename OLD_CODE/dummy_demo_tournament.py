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
def uniform_crossover_original(parents): 
    parent1, parent2 = np.hsplit(parents,2)
    section = np.random.uniform(size=parent1.shape)                         #It generates a random array roll of the same shape as parentsA 
                                                                            #and parentsB filled with random numbers from a uniform distribution 
                                                                            #between 0 and 1 using np.random.uniform.
    offspring = parent1 * (section>=0.5) + parent2 * (section < 0.5)                #it performs the crossover operation by selecting genes from either 
                                                                                    #parentsA or parentsB for each offspring based on the values in the 
                                                                                    # roll array. If the corresponding value in roll is greater than or 
                                                                                    # equal to 0.5, it selects the gene from parentsA, otherwise, 
                                                                                    # it selects the gene from parentsB
    #child2 = parent2 * (section>=0.5) + parent1 * (section < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return np.squeeze(offspring, 1)

# creates one child out of two parents, using uniform crossover
def uniform_crossover(parents): 
    parent1 = parents[0]
    parent2 = parents[1]
    section = np.random.uniform(size=parent1.shape)
    offspring = parent1 * (section>=0.5) + parent2 * (section < 0.5)
    # squeeze to get rid of the extra dimension created during parent selecting
    return offspring

# mutates the offspring using uniform mutation
def uniform_mutate_original(offspring, mutation_rate):
    # iterate over individuals in offspring
    for i in range(len(offspring)):
        # iterate over weights in individual
        for j in range(len(offspring[i])): 
            # with probability mutation_rate, alter weight in individual
            # with value sampled from uniform distribution.
            if np.random.uniform(0,1) <= mutation_rate: 
                offspring[i][j] = np.random.uniform(0, 1)
            
    return offspring  

def uniform_mutate(offspring, mutation_rate):
    for i in range(0, n_weights):
        if np.random.uniform(0,1) <= mutation_rate: 
            offspring[i] = np.random.uniform(0, 1)
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
    parents_original = parent_selection(pop, pop_fit, pop_size)
    offspring = []
    for j in range(int(pop_size/2)):
        parent1_index = tournament_selection(pop, pop_fit, 2)
        parent1 = pop[parent1_index]
        # print(parent1)
        parent2_index = tournament_selection(pop, pop_fit, 2)
        parent2 = pop[parent2_index]
            # print(parent2)
        parents = [parent1, parent2]
        # print(parents)
        offspring_individual = uniform_crossover(parents)
        offspring_individual = uniform_mutate(offspring_individual, mutation_rate)
        offspring.append(offspring_individual)
    offspring_fit = evaluate(offspring)
    pop = np.vstack((pop, offspring))                                                #we could also remove the entire old gen and use ordeirng to select the best 100 children
    pop_fit = np.concatenate([pop_fit, offspring_fit])
    pop, pop_fit = survivor_selection(pop, pop_fit, pop_size)
    print (f"Gen {i} - Best: {np.max (pop_fit)} - Mean: {np.mean(pop_fit)}")
# env.play()


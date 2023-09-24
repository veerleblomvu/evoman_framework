################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Group 65             #
#                              #
################################

# Import framwork and other libs
import sys
from evoman.environment import Environment
from demo_controller import player_controller
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random

# Choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Create a folder for the experiment in which all the data are stored
experiment_name = 'a_osd_war2_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# The NN has one hidden layer with 10 neurons
n_hidden_neurons = 10

# Initializes simulation in individual evolution mode, for single static enemy.
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

# OPTIMIZATION FOR CONTROLLER SOLUTION: GENETIC ALGORITHM

# Set time marker
ini = time.time()  

# Genetic algorithm parameters
run_mode = 'train' # train or test
n_weights = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5 # number of weights for multilayer with 10 hidden neurons
upper_bound = 1 # upper bound of start weights
lower_bound = -1 # lower bound of start weights
pop_size = 100
n_generations = 30
mutation_rate = 0.2
last_best = 0
sigma = 1
learning_rate = 0.1
boundary = 0.001
k = 2

def simulation(env,x):
    """
    Run a simulation for one individual in the population.

    Parameters:
        env (Environment): The environment object representing the game.
        pop (array-like): The controller or policy for the individual.

    Returns:
        float: The fitness score of the individual.
    """
    # Run the simulation and get fitness, player energy, enemy energy, and duration
    fit,p_energy,e_energy,duration = env.play(pcont=x)
    return fit

def normalize(x, pop_fit):
    """
    Normalize x based on a population fitnesses.

    Parameters:
        x (float): The value to be normalized.
        pop_fit (list): List of fitness scores in the population.

    Returns:
        float: The normalized value of 'x'.
    """
    if (max(pop_fit) - min(pop_fit)) > 0:
        x_norm = (x - min(pop_fit)) / (max(pop_fit) - min(pop_fit))
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001

    return x_norm

def evaluate(pop):
    """
    Determine the fitnesses of individuals in the population.

    Parameters:
        pop (list): The population of individuals.

    Returns:
        numpy.ndarray: An array containing the fitness score for each individual.
    """
    pop_fit = np.array(list(map(lambda y: simulation(env,y), pop)))
    return pop_fit

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

def uniform_mutate(offspring, mutation_rate, sigma):
    """
    Apply uniform mutation to the offspring.

    Args:
        offspring (numpy.ndarray): The offspring population.
        mutation_rate (float): The mutation rate.
        sigma (float): The sigma value.

    Returns:
        numpy.ndarray: The mutated offspring population.
    """
    for i in range(len(offspring)):
        if np.random.uniform(0,1) <= mutation_rate:
            offspring[i] = sigma * np.random.normal(0,1)
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

def survival_selection(parents, parents_fit, offspring, x, y):
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

def doomsday(pop, pop_fit):
    """
    Kills the worst genomes and replaces them with new best/random solutions.

    Parameters:
        pop (numpy.ndarray): The population of individuals.
        pop_fit (numpy.ndarray): The fitness scores of the individuals.

    Returns:
        tuple: A tuple containing the updated population and fitness scores.
    """
    # Select a quarter of the population as the worst performers
    ordered_pop_fit = np.argsort(pop_fit)
    worst_pop_fit = ordered_pop_fit[0:(int(pop_size/4))]

    # Iterte over the genomes of the worst-performing individuals in the population
    for i in worst_pop_fit:
        for j in range(0,n_weights):
            pro = np.random.uniform(0,1)
            # Replace it with a new random value 
            if np.random.uniform(0,1)  <= pro:
                pop[i][j] = np.random.uniform(lower_bound, upper_bound)
            # Replace it with the genetic component from the best individual in the population
            else:
                pop[i][j] = pop[ordered_pop_fit[-1:]][0][j] # dna from best

        # Evaluate the fitness of the new individual
        pop_fit[i]=evaluate([pop[i]])
    
    # Return the updated population
    return pop, pop_fit


# Loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# Initializes population loading old solutions or generating new ones
if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(lower_bound, upper_bound, (pop_size, n_weights))
    pop_fit = evaluate(pop)
    best = np.argmax(pop_fit)
    mean = np.mean(pop_fit)
    std = np.std(pop_fit)
    initial_gen = 0
    solutions = [pop, pop_fit]
    env.update_solutions(solutions)

else:

    print( '\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    pop_fit = env.solutions[1]

    best = np.argmax(pop_fit)
    mean = np.mean(pop_fit)
    std = np.std(pop_fit)

    # Find last generation number
    file_aux  = open(experiment_name+'/gen.txt','r')
    initial_gen = int(file_aux.readline())
    file_aux.close()

# Saves results for first population
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(initial_gen)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(initial_gen)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# EVOLUTIONARY LOOP

last_sol = pop_fit[best]
notimproved = 0

# Loop through generations
for i in range(initial_gen+1, n_generations):

    # Create offspring applying crossover and mutation
    offspring = whole_arithmic_crossover(pop, pop_fit, k, alpha=0.5)
    offspring = [uniform_mutate(gene, mutation_rate, sigma) for gene in offspring]
    sigma = update_sigma(sigma, learning_rate, boundary)
    
    # Survival selection (10 elite parents + 90 random children)
    pop = survival_selection(pop, pop_fit, offspring, 10, 90)
    pop_fit = evaluate(pop)

    # Get max fitness in new population
    best_sol = np.max(pop_fit)

    # Searching new areas
    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        # Write to a results file
        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        # Perform doomsday operation
        pop, pop_fit = doomsday(pop,pop_fit)
        notimproved = 0

    best = np.argmax(pop_fit)
    std  =  np.std(pop_fit)
    mean = np.mean(pop_fit)


    # Saves result
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(pop_fit[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
    file_aux.close()

    # Save generation number
    file_aux  = open(experiment_name+'/gen.txt','w')
    file_aux.write(str(i))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt',pop[best])

    # Save simulation state
    solutions = [pop, pop_fit]
    env.update_solutions(solutions)
    env.save_state()

fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()

env.state_to_log() # checks environment state

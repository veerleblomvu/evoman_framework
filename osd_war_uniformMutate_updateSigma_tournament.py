###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import time
import numpy as np
from math import fabs,sqrt
import glob, os
import random

# choose this for not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'a_osd_war2_1'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

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

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 100
gens = 30
mutation = 0.2
last_best = 0
sigma = 1
learning_rate = 0.1
boundary = 0.001
k = 2

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# normalizes
def norm(x, pfit_pop):

    if ( max(pfit_pop) - min(pfit_pop) ) > 0:
        x_norm = ( x - min(pfit_pop) )/( max(pfit_pop) - min(pfit_pop) )
    else:
        x_norm = 0

    if x_norm <= 0:
        x_norm = 0.0000000001
    return x_norm


# evaluation
def evaluate(x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


# tournament
def tournament_karine(pop):
    c1 =  np.random.randint(0,pop.shape[0], 1)
    c2 =  np.random.randint(0,pop.shape[0], 1)

    if fit_pop[c1] > fit_pop[c2]:
        return pop[c1][0]
    else:
        return pop[c2][0]
    
    
def tournament(pop, k):
    number_individuals = pop.shape[0]
    current_winner = np.random.randint(0, number_individuals-1)
    score = fit_pop[current_winner]
    for candidates in range(k-1): #We already have one candidate, so we are left with k-1 to choose
        contender_number = np.random.randint(0, number_individuals-1)
        if fit_pop[contender_number] > score:
            current_winner = contender_number
            score = fit_pop[contender_number]
    return pop[current_winner]
    
def tournament_old2(pop,k):
    n_individuals = pop.shape[0]
    samples = random.sample(range(0, n_individuals-1), k)
    pop_fit = evaluate(pop)
    sample_fitnesses = pop_fit[samples]
    winner = np.argmax(sample_fitnesses)
    return pop[winner]

# limits
def limits(x):
    if x>dom_u:
        return dom_u
    elif x<dom_l:
        return dom_l
    else:
        return x


#crossover - old function where crossover and mutation were together
def crossover(pop):

    total_offspring = np.zeros((0,n_vars))

    for p in range(0,pop.shape[0], 2):
        p1 = tournament(pop)
        p2 = tournament(pop)

        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring =  np.zeros( (n_offspring, n_vars) )

        for f in range(0,n_offspring):

            cross_prop = np.random.uniform(0,1)
            offspring[f] = p1*cross_prop+p2*(1-cross_prop)

            # mutation for 1 individial 
            for i in range(0,len(offspring[f])):
                if np.random.uniform(0 ,1)<=mutation:
                    offspring[f][i] =   offspring[f][i]+np.random.normal(0, 1)

            offspring[f] = np.array(list(map(lambda y: limits(y), offspring[f])))

            total_offspring = np.vstack((total_offspring, offspring[f]))

    return total_offspring

#crossover - old function, but splitted
def karine_crossover(pop):
    total_offspring = np.zeros((0,n_vars))
    for p in range(0,pop.shape[0], 2):
        p1 = pop[0]
        p2 = pop[1]
        n_offspring =   np.random.randint(1,3+1, 1)[0]
        offspring =  np.zeros( (n_offspring, n_vars))

        for f in range(0,n_offspring):
            cross_prop = np.random.uniform(0,1)
            offspring[f] = p1*cross_prop+p2*(1-cross_prop)
            total_offspring = np.vstack((total_offspring, offspring[f]))
    return total_offspring

def whole_arithmic_crossover(pop, alpha=0.5):
    offspring = []
    for p in range(0, pop.shape[0], 2):
        parent1 = tournament(pop, k)
        parent2 = tournament(pop, k)
        child1 = []
        child2 = []
        for gene1, gene2 in zip(parent1, parent2):
            offspring1 = alpha * gene1 + (1 - alpha) * gene2
            offspring2 = alpha * gene2 + (1 - alpha) * gene1
            child1.append(offspring1)
            child2.append(offspring2)
        offspring.extend([child1, child2])
    return np.array(offspring)

def karine_mutation(offspring):
    # mutation for 1 individial 
    for i in range(0,len(offspring)):
        randomnr = np.random.uniform(0 ,1)
        #print("random", randomnr)
        if randomnr <= mutation:
            random2 = np.random.normal(0, 1)
            #print("random2", random2)
            offspring[i] =   offspring[i]+random2
            #print("offspring[1]", offspring[i])
        #print("offspring", offspring)
    offspring = np.array([limits(y) for y in offspring])
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

def uniform_mutate(offspring):
    """takes the offspring and mutates it
    Args: offspring (list)
    """
    # mutation for 1 individual
    for i in range(len(offspring)):
        if np.random.uniform(0,1) <= mutation:
            offspring[i] = sigma * np.random.normal(0,1)
    offspring = np.array([limits(y) for y in offspring])
    return offspring

# kills the worst genomes, and replace with new best/random solutions
def doomsday(pop,fit_pop):

    worst = int(npop/4)  # a quarter of the population
    order = np.argsort(fit_pop)
    orderasc = order[0:worst]

    for o in orderasc:
        for j in range(0,n_vars):
            pro = np.random.uniform(0,1)
            if np.random.uniform(0,1)  <= pro:
                pop[o][j] = np.random.uniform(dom_l, dom_u) # random dna, uniform dist.
            else:
                pop[o][j] = pop[order[-1:]][0][j] # dna from best

        fit_pop[o]=evaluate([pop[o]])

    return pop,fit_pop



# loads file with the best solution for testing
if run_mode =='test':

    bsol = np.loadtxt(experiment_name+'/best.txt')
    print( '\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed','normal')
    evaluate([bsol])

    sys.exit(0)


# initializes population loading old solutions or generating new ones

if not os.path.exists(experiment_name+'/evoman_solstate'):

    print( '\nNEW EVOLUTION\n')

    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))
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
file_aux  = open(experiment_name+'/results.txt','a')
file_aux.write('\n\ngen best mean std')
print( '\n GENERATION '+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
file_aux.write('\n'+str(ini_g)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
file_aux.close()


# evolution

last_sol = fit_pop[best]
notimproved = 0

for i in range(ini_g+1, gens):

    offspring = whole_arithmic_crossover(pop)  # recombination
    offspring = [uniform_mutate(gene) for gene in offspring]
    sigma = update_sigma(sigma, learning_rate)

    #offspring = [karine_mutation(gene) for gene in offspring]  #because recom and mut are splitted
    #offspring = crossover(pop)  # crossover --> original way 
    fit_offspring = evaluate(offspring)   # evaluation
    pop = np.vstack((pop,offspring))
    fit_pop = np.append(fit_pop,fit_offspring)

    best = np.argmax(fit_pop) #best solution in generation
    fit_pop[best] = float(evaluate(np.array([pop[best] ]))[0]) # repeats best eval, for stability issues
    best_sol = fit_pop[best]

    # selection
    fit_pop_cp = fit_pop
    fit_pop_norm =  np.array(list(map(lambda y: norm(y,fit_pop_cp), fit_pop))) # avoiding negative probabilities, as fitness is ranges from negative numbers
    probs = (fit_pop_norm)/(fit_pop_norm).sum()
    chosen = np.random.choice(pop.shape[0], npop , p=probs, replace=False)
    chosen = np.append(chosen[1:],best)
    pop = pop[chosen]
    fit_pop = fit_pop[chosen]


    # searching new areas

    if best_sol <= last_sol:
        notimproved += 1
    else:
        last_sol = best_sol
        notimproved = 0

    if notimproved >= 15:

        file_aux  = open(experiment_name+'/results.txt','a')
        file_aux.write('\ndoomsday')
        file_aux.close()

        pop, fit_pop = doomsday(pop,fit_pop)
        notimproved = 0

    best = np.argmax(fit_pop)
    std  =  np.std(fit_pop)
    mean = np.mean(fit_pop)


    # saves results
    file_aux  = open(experiment_name+'/results.txt','a')
    print( '\n GENERATION '+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6)))
    file_aux.write('\n'+str(i)+' '+str(round(fit_pop[best],6))+' '+str(round(mean,6))+' '+str(round(std,6))   )
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




fim = time.time() # prints total execution time for experiment
print( '\nExecution time: '+str(round((fim-ini)/60))+' minutes \n')
print( '\nExecution time: '+str(round((fim-ini)))+' seconds \n')


file = open(experiment_name+'/neuroended', 'w')  # saves control (simulation has ended) file for bash loop file
file.close()


env.state_to_log() # checks environment state

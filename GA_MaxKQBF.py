import time
from datetime import datetime
import random

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt


#############################################################################
def read_data(data_path, show=False):
    """Read the specified data (KQBF problem)"""
    # data_path = "./instances/kqbf/kqbf020"

    with open(data_path, "r") as file:
        lines = file.readlines()

    N = int(lines[0].strip())
    W = int(lines[1].strip())
    w = list(map(int, lines[2].strip().split()))
    A = []

    for i in range(N):
        row = list(map(int, lines[i + 3].strip().split()))
        A.append(row)

    max_length = max(len(sublist) for sublist in A)
    for sublist in A:
        sublist.extend([0] * (max_length - len(sublist)))

    matrix = np.zeros((max_length, max_length))
    for i in range(max_length):
        for j in range(i + 1):
            matrix[i][j] = A[i][j]
            matrix[j][i] = A[i][j]

    if show:
        print("N:", N)
        print("W:", W)
        print("w:", w)
        print("A:")
        for row in A:
            print(row)

    return N, W, np.array(w), np.array(matrix)

def write_to_csv(t_inspection, minutes, best_solution, best_fitness, config_name, filename="./outputs/csv/results.csv"):
    # The 'a' parameter in open function stands for 'append'
    # It will create the file if it doesn't exist, and append to it if it does
    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        # Check if file is empty
        if file.tell() == 0:
            # Write the header
            writer.writerow(["Time", "RunningTime", "Best Solution", "Best Fitness", "Config Name"])
        # Write the data
        writer.writerow([t_inspection, minutes, best_solution, best_fitness, config_name])

def str_to_bool(v):

    if v.lower() in ("sim", "true", "1"):
        return True
    elif v.lower() in ("nao", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean type expected not passed")

#############################################################################
# Definindo a função objetivo QBF
def objective_function(x, A):
    return np.dot(x, np.dot(A, x.T))

# Definindo a restrição do problema
def constraint(x, w, W, show=False):
    result = np.dot(x, w)
    if show:
        print(f"<x|w> <= W?")
        print(f"<{x}|{w}> = {result} \n <= {W}??? {result <= W}")
    return result <= W

#############################################################################
def initialize_population(population_size, n):
    """Initialize the population using a random strategy"""
    population = []
    for _ in range(population_size):

        x = np.random.randint(2, size=n)
        population.append(x)

    return population

# Latin Hypercube
def lhs_sample(n, dims):
    """
    Generate Latin Hypercube Samples (LHS) for given dimensions and number of samples.

    Parameters:
    n (int): Number of samples.
    dims (int): Number of dimensions.

    Returns:
    np.ndarray: Latin Hypercube Samples.
    """
    # Inicializa o array de amostras
    lhs_samples = np.empty((n, dims))

    # Divide o intervalo em n partes iguais e escolhe um valor em cada intervalo
    for i in range(dims):
        lhs_samples[:, i] = np.random.uniform(size=n)

    # Embaralha as amostras em cada dimensão
    for i in range(dims):
        np.random.shuffle(lhs_samples[:, i])

    return lhs_samples

def initialize_population_lhs(population_size, n):
    """
    Initialize the population using Latin Hypercube Sampling.

    Parameters:
    population_size (int): Size of the population.
    n (int): Number of dimensions.

    Returns:
    list: Initial population using Latin Hypercube Sampling.
    """
    population = []
    lhs_samples = lhs_sample(population_size, n)  # Gerar várias amostras LHS
    for sample in lhs_samples:
        x = (sample > 0.5).astype(int)  # Convertendo valores em binário
        population.append(np.array(x))  # Convertendo para numpy array
    return population

def stochastic_universal_selection(population, fitness_values, n_parents):
    # Calcula a soma total das pontuações de fitness
    total_fitness = sum(fitness_values)
    
    # Calcula a distância entre cada segmento
    segment_distances = total_fitness / n_parents if total_fitness != 0 else 1
    
    # Gera um ponto de partida aleatório dentro do primeiro segmento
    start_point = np.random.uniform(0, segment_distances)
    
    # Inicializa os índices dos pais selecionados
    selected_parents = []

    # # TEST #
    # pointers = [start_point + i*segment_distances for i in range(n_parents)]

    # for pointer in pointers:
    #     i = 0
    #     while sum(fitness_values[:i+1]) < pointer:
    #         i += 1
    #     selected_parents.append(population[i])
    # # END TEST #
    
    # Inicializa o ponto de partida para a seleção
    selected_point = start_point

    # Seleciona os pais
    while len(selected_parents) < n_parents:
        # Percorre a população para encontrar o indivíduo correspondente ao ponto selecionado
        cumulative_fitness = 0
        prev_len = len(selected_parents)
        for idx, ind_fitness in enumerate(fitness_values):
            cumulative_fitness += ind_fitness
            # Se a soma cumulativa das pontuações de fitness for maior que o ponto selecionado,
            # adiciona o indivíduo correspondente à lista de pais selecionados

            if cumulative_fitness > selected_point:
                selected_parents.append(population[idx])
                # Atualiza o ponto selecionado para o próximo segmento
                selected_point += segment_distances
                break


        # If no parent was selected in this iteration, select a random parent
        if len(selected_parents) == prev_len:
            selected_parents.append(random.choice(population))
    
    return selected_parents

def uniform_crossover(parent1, parent2):
    # Inicializa o filho com o mesmo formato que os pais
    child = np.empty_like(parent1)
    
    # Realiza o cruzamento
    for i in range(len(parent1)):
        # Seleciona aleatoriamente o gene do pai 1 ou pai 2 para o filho
        if np.random.rand() < 0.5:  # 50% de chance de selecionar o gene do pai 1
            child[i] = parent1[i]
        else:
            child[i] = parent2[i]
    
    return child

def mutation(bitstring, mutation_rate):
	for i in range(len(bitstring)):
		# check for a mutation
		if np.random.rand() < mutation_rate:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]

#############################################################################
def genetic_algorithm(config_name,
                      A, w, W, 
                      population_size, 
                    #   number_generation, 
                      minutes,
                      lhs_inicialization=False,
                      su_selection=False,
                      unif_crossover=False,
                      dynamic_mutation=True,
                      mutation_rate=.6):
    
    print(f"GA with configs: {config}")

    # Initialization

    n = len(w)
    if lhs_inicialization:
        population = initialize_population_lhs(population_size, n)
    else:
        population = initialize_population(population_size, n) 
    fitness_history = []

    best_solution = None
    best_fitness = float("-inf")

    # for generation in range(number_generation):
    secs = minutes * 60
    start_time = time.time()
    time_ = time.time() - start_time
    while (time_) < secs:
        # print(f"Time: {time_}")

        generation = time_ * 60

        # Evaluation
        fitness = [objective_function(x, A) for x in population]

        feasible_population = [x for x in population if constraint(x, w, W)]

        # Selection 

        if feasible_population:

            feasible_fitness = [objective_function(x, A) for x in feasible_population]
            # Selecting parents with the highest feasible fitness
            if su_selection:
                parents = stochastic_universal_selection(population, feasible_fitness, population_size)
            else:
                parents = random.choices(feasible_population, weights=feasible_fitness, k=population_size)
            
        else:

            parents = []
            while len(parents) < population_size:
                # Restart the initial population until we find a feasible solution
                if lhs_inicialization:
                    adjusted_population = initialize_population_lhs(population_size, n)
                else:
                    adjusted_population = initialize_population(population_size, n)
                if any(weight > 0 for weight in fitness):
                    if su_selection:
                        potential_parents = stochastic_universal_selection(adjusted_population, fitness, population_size)
                    else:
                        potential_parents = random.choices(adjusted_population, weights=fitness)
                    
                else:
                    potential_parents = random.choices(adjusted_population)

                try:
                    if constraint(potential_parents, w, W):
                        parents.append(potential_parents)
                except:
                    continue
                
        offspring = []
        for _ in range(population_size):
            
            parent1, parent2 = random.sample(parents, 2)
            
            if unif_crossover:
                child = uniform_crossover(parent1, parent2)
            else:
                child = np.array([random.choice([bit1, bit2]) for bit1, bit2 in zip(parent1, parent2)])
            

            offspring.append(child)
        
        # Mutation

        if dynamic_mutation == True:
            dynamic_mutation_rate = 1 / (generation + 1) # Dynamic mutation rate
            for i in range(population_size):

                if random.random() < dynamic_mutation_rate:

                    mutation_index = random.randint(0, len(offspring[i])-1)
                    offspring[i][mutation_index] = 1 - offspring[i][mutation_index]  # Flip the bit at the mutation_index
        else:
            for i in range(population_size): # Fixed mutation rate
                mutation(offspring[i], mutation_rate)
    
        # Elitism

        if best_solution is not None:
            offspring[0] = best_solution # Best solution is preserved in next generation

        population = offspring

        # Find the best feasible solution
        feasible_solutions = [x for x in population if constraint(x, w, W)]

        if feasible_solutions:

            best_solution = max(feasible_solutions, key=lambda x: objective_function(x, A))

            best_fitness = objective_function(best_solution, A)

            try:
                best_fitness = float(np.sum(objective_function(best_solution, A)))
            except:
                pass

        fitness_history.append(best_fitness)

        time_ = time.time() - start_time
        if time_ < 1e-5:
            break

    # Plot fitness progress
    plt.plot(range(1, len(fitness_history)+1), fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("Genetic Algorithm - Fitness progress")

    # Save the figure before showing it
    plt.savefig(f'./outputs/plots/{config}_fitness_progress.png', dpi=300)

    return best_solution, best_fitness


#############################################################################
if __name__ == "__main__":

    # HOW TO RUN? Call inside a terminal: 
    # python3 GA_MaxKQBF.py kqbf020 --pop-size 50 --minutes 30

    # Using minutes as counter
    print()
    print("Reading the data...")

    parser = argparse.ArgumentParser(description="Process the problem instance.")
    parser.add_argument("problem_instance", type=str, help="Problem instance string")
    parser.add_argument("--pop-size", type=int, default=100, help="Population size (default: 100)")

    parser.add_argument("--lhs", type=str_to_bool, default=False, help="Configuration used to solve the problem")
    parser.add_argument("--su", type=str_to_bool, default=False, help="Configuration used to solve the problem")
    parser.add_argument("--crossover", type=str_to_bool, default=False, help="Configuration used to solve the problem")
    parser.add_argument("--dmutation", type=str_to_bool, default=True, help="Configuration used to solve the problem")
    parser.add_argument("--mutation-rate", type=float, default=0.5, help="Configuration used to solve the problem")

    parser.add_argument("--minutes", type=float, default=1, help="Duration in minutes (default: 1)")

    args = parser.parse_args()

    problem_instance = args.problem_instance
    
    pop_size = args.pop_size
    lhs_init = bool(args.lhs)
    sus = bool(args.su) 
    cross = bool(args.crossover)
    dyn_mut = bool(args.dmutation)
    rate = args.mutation_rate

    minutes = args.minutes

    N, W, w, A = read_data(data_path=f"./instances/kqbf/{problem_instance}", show=False)
    w = np.array(w)

    config = f"LHS={lhs_init}_SUS={sus}_UNIF_CROSS={cross}_DM={dyn_mut}_DRATE={rate}_PSIZE={pop_size}"
    now = datetime.now()
    print()
    print(f"Starting the search at {now}")
    print(f"...for the instance {problem_instance} with config {config}...")
    print()

    # Aplicando o algoritmo genético
    best_solution, best_fitness = genetic_algorithm(config,
                                                    A, w, W, 
                                                    pop_size, 
                                                    #   number_generation, 
                                                    minutes,
                                                    lhs_inicialization=lhs_init,
                                                    su_selection=sus,
                                                    unif_crossover=cross,
                                                    dynamic_mutation=dyn_mut,
                                                    mutation_rate=rate)
    
    write_to_csv(now, minutes, best_solution, best_fitness, config)

    print(f"Best solution: {best_solution}")
    print(f"Best fitness: {best_fitness}")
    print()
    print("Finished the process...")



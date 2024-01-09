import random, time

# Define the PID parameters range

from progress.bar import IncrementalBar

from tqdm import tqdm


import serial
import re

def extract_values(message):
    pattern = r'\d+'
    match = re.findall(pattern, message)
    
    group_calc, g_calc, group_real, g_real = match

    return (group_calc, g_calc, group_real, g_real)
        

def send_and_receive_data(serial_port, command):
    serial_port.write(command.encode())
    response = serial_port.read_until(b'EC\rNC\r').decode()
    return response

# Replace 'COM1' with the actual serial port name
serial_port_name = 'COM6'
baud_rate = 115200

ser = serial.Serial(serial_port_name, baud_rate, timeout=0.1)

command = f"AC2\r"
ser.write(command.encode())
response = ser.read_until(b'NC\r').decode()

best_individuals_L = []
best_individuals_R = []

current_timeout_penalty = 0


# Function to simulate the PID control and return the number of steps
def simulate_pid(right_p, right_i, right_d, left_p, left_i, left_d, end_flag=False):
    global current_timeout_penalty

    try:
        command = f"DP{right_p}\r"
        ser.write(command.encode())
        response = ser.read_until(b'NC\r').decode()

        command = f"GP{left_p}\r"
        ser.write(command.encode())
        response = ser.read_until(b'NC\r').decode()

        command = f"CI{(right_i + left_i) / 2}\r"
        ser.write(command.encode())
        response = ser.read_until(b'NC\r').decode()
       
        # print(f"Send: Left PD: {left_p} {left_d} Right PD:{right_p} {right_d}")

        ser.timeout = 1.2

        command = "av50\r"

        time_before = time.perf_counter()
        ser.write(command.encode())
        response = ser.read_until(b'NC\r').decode()

        time_taken = time.perf_counter() - time_before

        if time_taken > 1000:
            current_timeout_penalty = 50

        ser.timeout = 0.1

        
        if len(response) < 1:
            command = "ST\r"
            ser.write(command.encode())
            response = ser.read_until(b'NC\r\n').decode()

    

        

        command = "ec\r"
        response = send_and_receive_data(ser, command)

        if end_flag:
            print(f"Got : {response}")

        # Assuming the values are directly available in the response
        g_calc, g_real, d_calc, d_real = extract_values(response)

        if end_flag:
            print(g_calc, g_real, d_calc, d_real)

        

    
        command = "re50\r"

        ser.write(command.encode())
        response = ser.read_until(b'NC\r').decode()

        return (int(g_real), int(d_real))
            
        
        
    except serial.SerialException as e:
        print(f"Error opening the serial port: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    

import random

# Define the PID parameters range
P_range = (1400, 1900)
I_range = (0, 0)
D_range = (900, 1100)

# Define the population size
population_size = 20

current_population_size = population_size

# Define the number of generations
num_generations = 18

# Function to calculate the fitness based on the actual steps
def calculate_fitness(target, actual):
    return 1 / (0.01 + abs(target - actual) + current_timeout_penalty)

# Function to create an initial population
def generate_population():
    population_R = []
    population_L = []
    for _ in range(0, population_size):
        # print(_)

        right_p = random.uniform(*P_range)
        right_i = random.uniform(*I_range)
        right_d = random.uniform(*D_range)
        
        left_p = random.uniform(*P_range)
        left_i = random.uniform(*I_range)
        left_d = random.uniform(*D_range)

        while((abs(left_p - right_p) > 30) or (abs(left_d - right_d) > 30)):
            right_p = random.uniform(*P_range)
            right_i = random.uniform(*I_range)
            right_d = random.uniform(*D_range)
            
            left_p = random.uniform(*P_range)
            left_i = random.uniform(*I_range)
            left_d = random.uniform(*D_range)
        

        
        population_R.append((right_p, right_i, right_d))
        population_L.append((left_p, left_i, left_d))
    return (population_L, population_R)



# Function to mutate an individual
def mutate(individual):
    mutated_individual = list(individual)
    for i in range(3):
        if i != 1:
            if random.random() < 0.72:  # Mutation probability
                mutated_individual[i] += random.uniform(-50, 50)  # Adjust mutation range as needed
                if i % 3 == 0 or i % 3 == 1:  # Make sure P and I values are within bounds
                    mutated_individual[i] = max(min(mutated_individual[i], P_range[1]), P_range[0])
                else:  # D values are within bounds
                    mutated_individual[i] = max(min(mutated_individual[i], D_range[1]), D_range[0])

    return tuple(mutated_individual)

# Function to select individuals for the next generation
def select(population, fitness_scores):
    global current_population_size

    # print(len(population), len(fitness_scores))

    selected = random.choices(population, weights=fitness_scores, k=(current_population_size-1))

    # print(len(population), len(selected))

    return selected

# Main genetic algorithm
def pid_tuning_genetic_algorithm():
    global current_population_size

    population_L, population_R = generate_population()


    for generation in tqdm(range(num_generations), desc="Generations"):
        # print(f"Generation {generation + 1}")

        target_steps = 101

        fitness_scores_L = []
        fitness_scores_R = []

        print("")
        individual_progress = IncrementalBar("Mesuring individual fitness", max=current_population_size)

        for individual_index in range(0, current_population_size):
            # print(individual_index)


            # Evaluate fitness for each individual in the population
            REAL_L, REAL_R = simulate_pid(*population_R[individual_index], *population_L[individual_index])

            fitness_scores_L.append(calculate_fitness(target_steps, REAL_L))
            fitness_scores_R.append(calculate_fitness(target_steps, REAL_R))

            individual_progress.next()

        individual_progress.finish()



        
        # print(population_L)
        # print(population_R)


        # print(fitness_scores_L)
        # print(fitness_scores_R)

        # Select individuals for the next generation
        selected_population_L = select(population_L, fitness_scores_L)
        selected_population_R = select(population_R, fitness_scores_R)


        # print(selected_population_L)
        # print(selected_population_R)

        # Display the best individual in the current generation
        best_individual_L = population_L[fitness_scores_L.index(max(fitness_scores_L))]
        best_individual_R = population_R[fitness_scores_R.index(max(fitness_scores_R))]

        # Apply crossover and mutation
        new_population_L = [mutate(random.choice(selected_population_L)) for _ in range(0, current_population_size-1)]
        new_population_R = [mutate(random.choice(selected_population_R)) for _ in range(0, current_population_size-1)]

        # Replace the old population with the new one
        population_L = new_population_L
        population_R = new_population_R


        print("Best PID parameters:")
        print(f"For left: {best_individual_L}")
        print(f"For right: {best_individual_R}")

        best_individuals_L.append(best_individual_L)
        best_individuals_R.append(best_individual_R)

        print("Testing G/D calc vs. real: ")
        simulate_pid(*best_individual_R, *best_individual_L, end_flag=True)

        current_population_size -= 1


    best_individuals_fitness_L = []
    best_individuals_fitness_R = []

    final_bar_individual = IncrementalBar("Selecting the best of the bests", max=len(best_individuals_L))
    for individual_index in range(0, len(best_individuals_L)):
            # print(individual_index)


            # Evaluate fitness for each individual in the population
            print(best_individuals_R[individual_index], best_individuals_L[individual_index])
            REAL_L, REAL_R = simulate_pid(*best_individuals_R[individual_index], *best_individuals_L[individual_index], end_flag=True)

            best_individuals_fitness_L.append(calculate_fitness(target_steps, REAL_L))
            best_individuals_fitness_R.append(calculate_fitness(target_steps, REAL_R))

            final_bar_individual.next()

    best_of_bests_individual_L = best_individuals_L[best_individuals_fitness_L.index(max(best_individuals_fitness_L))]
    best_of_bests_individual_R = best_individuals_R[best_individuals_fitness_R.index(max(best_individuals_fitness_R))]
    
    print("\n\n")

    print("True Best PID parameters:")
    print(f"For left: {best_of_bests_individual_L}")
    print(f"For right: {best_of_bests_individual_R}\n")

    print("Testing G/D calc vs. real: ")
    simulate_pid(*best_of_bests_individual_R, *best_of_bests_individual_L, end_flag=True)

    final_bar_individual.finish()






# Run the genetic algorithm
pid_tuning_genetic_algorithm()
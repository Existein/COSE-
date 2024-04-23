import sys
import subprocess
import requests
import time
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
import matplotlib as mtp
import numpy as np
import pandas as pd
from random import *
import matplotlib.pyplot as plt
# from matplotlib import gridspec
# from mpl_toolkits.mplot3d import Axes3D
seed(5000)

def get_cost(x, y):
    # Read the CSV file
    df = pd.read_csv('io_benchmark_results.csv')

    # Find the matching pairs of x and y
    matching_rows = df[(df['Memory (MiB)'] == x) & (df['vCPUs'] == y)]

    # If there are no matching rows, return None
    if matching_rows.empty:
        print(f"No matching rows found for memory={x} and vcpu={y}")
        return None

    # Get the execution time and cost
    exec_time = matching_rows['Avg. Execution Time'].values[0]
    cost = matching_rows['Cost ($)'].values[0]

    # return exec_time, cost
    return cost

def next_finder (x, y):
    CONFIGURATIONS = [
    (128, 0.083), (128, 0.167), (128, 0.333), (128, 0.583), (128, 1), (128, 2),
    (256, 0.083), (256, 0.167), (256, 0.333), (256, 0.583), (256, 1), (256, 2),
    (512, 0.083), (512, 0.167), (512, 0.333), (512, 0.583), (512, 1), (512, 2),
    (1024, 0.583), (1024, 1), (1024, 2), 
    (2048, 1), (2048, 2), (2048, 4),
    (4096, 1), (4096, 2), (4096, 4), (4096, 6), (4096, 8), 
    (8192, 2), (8192, 4), (8192, 6), (8192, 8), 
    (16384, 4), (16384, 6), (16384, 8)
    ]

    # Find min and max values for normalization
    max_memory = max(config[0] for config in CONFIGURATIONS)
    min_memory = min(config[0] for config in CONFIGURATIONS)
    max_vCPUs = max(config[1] for config in CONFIGURATIONS)
    min_vCPUs = min(config[1] for config in CONFIGURATIONS)  

    # Normalize
    norm_memory = (x - min_memory) / (max_memory - min_memory)
    norm_vCPUs = (y - min_vCPUs) / (max_vCPUs - min_vCPUs)

    # Distance calculation with normalized values
    distances = [
        np.linalg.norm(
            np.array([
                (config[0] - min_memory) / (max_memory - min_memory), 
                (config[1] - min_vCPUs) / (max_vCPUs - min_vCPUs)
            ]) - np.array([norm_memory, norm_vCPUs])
        )
        for config in CONFIGURATIONS  
    ]

    closest_index = np.argmin(distances)
    
    memory, vCPUs = CONFIGURATIONS[closest_index]
    return {'x': memory, 'y': vCPUs}

if __name__ == "__main__":

    # define exploration space i.e. memory values 
    pbounds = { 'x': (128, 16384), 'y': (0.083, 8)}

    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=randint(128, 16384)
    )

    # BO parameters 
    acq_type = "ei"
    xi_value = 0.001 # 0.001 for more exploration
    kappa_value = 10.0
    alpha_value = 0.1
    utility = UtilityFunction(kind=acq_type, kappa = kappa_value, xi=xi_value)

    # Plot parameters
    plt.rcParams.update({'font.size': 10})  

    # BO in action ...
    counter = 1
    total_samples = 96

    f = open('xi_0001_output.txt', 'w')

    init_list = [ {'x': 128, 'y': 0.083}, {'x': 512, 'y': 0.333}, {'x': 4096, 'y': 2}, {'x': 16384, 'y': 8} ]
    f.write("Initial points probing...\n")
    for j in range(4):
        next_point = init_list[j]
        f.writelines(["Next point to probe is: " + repr(next_point) + "\n"])
        target = 1 / get_cost(**next_point)
        f.writelines(["Found the target value to be: " + repr(1 / target) + "\n"])
        next_point['x'] = next_point['x'] + random()
        f.writelines(["After randomized: " + repr(next_point) + "\n"])
        optimizer.register(params=next_point, target=target)

    f.write("\nMain phase initiated...\n")

    prev_ei = 1

    while True:
        suggested_point = optimizer.suggest(utility)
        suggested_point_array = np.array([[suggested_point['x'], suggested_point['y']]]) # Convert the dictionary to a 2D array
        # mu, sigma = optimizer._gp.predict(suggested_point_array, return_std=True)
        y_max = optimizer.max['target'] # not inverted!
        ei = utility.utility(suggested_point_array, optimizer._gp, y_max)
        if counter > 1: 
            f.writelines(['EI/PrevEI:', repr(ei/prev_ei), '\n'])
        if (ei/prev_ei) > -1.05 or counter == 1:
            next_point = next_finder(**suggested_point)
            f.writelines(["Next point to probe is:", repr(next_point), '\n'])
            target = 1 / get_cost(**next_point)
            f.writelines(["Found the target value to be:", repr(1 / target), '\n'])  
            next_point['x'] = next_point['x'] + random()
            f.writelines(["After randomized: ", repr(next_point), '\n'])  
            optimizer.register(params=next_point, target=target)
            optimizer.set_gp_params(alpha=alpha_value)
            optimizer.set_gp_params(normalize_y=True)
        else: # do not accept the suggestion
            # f.write("Use previous suggestion\n")
            # f.writelines(["Found the target value to be:", repr(1 / target), '\n'])
            # f.writelines(["After randomized: ", repr(next_point), '\n'])
            f.write("Use the best suggestion\n")
            f.writelines(["Found the target value to be:", repr(1 / optimizer.max['target']), '\n'])
            f.writelines(["After randomized: ", repr(optimizer.max['params']), '\n'])

        prev_ei = ei # for ei comparison

        f.writelines(['Samples so far:', repr(counter), '\n'])
        counter+=1
        if counter > total_samples:
            break
        
    f.writelines(["BEST:", str(1 / optimizer.max['target']), str(optimizer.max['params']), '\n\n'])

    f.close()
    
    # Initialize lists to store the data
    sample = []
    memory = []
    vcpu = []
    cost = []

    # Read the file and extract data
    with open('xi_0001_output.txt', 'r') as f:
        lines = f.readlines()
        current_sample = 1  

        for line in lines:
            if "After randomized:" in line:
                parts = line.split("{")[1].split("}")[0].split(",")
                memory.append(float(parts[0].split(":")[1]))
                vcpu.append(float(parts[1].split(":")[1]))
                sample.append(current_sample)
                current_sample += 1  
            elif "Found the target value to be:" in line:
                cost.append(float(line.split(":")[1]))

    # Create DataFrames
    df_3d = pd.DataFrame({'sample': sample, 'memory': memory, 'vcpu': vcpu})
    df_2d = pd.DataFrame({'sample': sample, 'cost': cost})

    # Create subplots for the two plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), dpi=300) 

    # 2D scatter plot (Sample, Memory, vCPU)
    scatter = ax1.scatter(df_3d['sample'], df_3d['memory'], c=df_3d['vcpu'], s=20, alpha=0.8, edgecolor='black', cmap='viridis')
    # ax1.set_yticks([128, 256, 512, 1024, 2048, 4096, 8192, 16384])
    # ax1.set_yticklabels(['128', '256', '512', '1024', '2048', '4096', '8192', '16384'])
    ax1.set_yticks([100, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000])
    ax1.set_yticklabels(['100', '500', '1000', '2000', '4000', '6000', '8000', '10000', '12000', '14000', '16000'])
    ax1.set_aspect(0.006)
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Memory (MiB)') 
    

    # 2D scatter plot (Memory vs. Cost)
    ax2.scatter(df_2d['sample'], df_2d['cost'], s=20, edgecolor='black', color='royalblue')
    # ax2.set_yscale('log')
    ax2.set_yticks([3e-5, 4e-5, 5e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4])
    ax2.set_yticklabels(['3e-5', '4e-5', '5e-5', '1e-4', '2e-4', '3e-4', '4e-4', '5e-4'])
    ax2.set_aspect('auto')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Cost ($)')
    

    # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.2)
    plt.colorbar(scatter, label='vcpu')
    plt.savefig('xi_0001_output.png')
    plt.close()

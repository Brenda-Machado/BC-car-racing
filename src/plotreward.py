"""
Brenda Silva Machado - 21101954
2024/2

"""

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

def plot_reward(path):
    pkl_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    plt.figure(figsize=(10, 6))

    for i, file in enumerate(pkl_files):
        file_path = os.path.join(path, file)
        with open(file_path, 'rb') as f:
            fitness_data = pickle.load(f)
            print(fitness_data)
        
        plt.plot(fitness_data, label=f'Model of {file.split(".")[0]}')

    plt.xlabel('Episodes/Trajectories')
    plt.ylabel('Rewards')
    plt.legend()  
    plt.show()


plot_reward('src/data/reward/') 

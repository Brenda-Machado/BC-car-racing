"""
Brenda Silva Machado - 21101954
2024/2

"""

import pickle
import matplotlib.pyplot as plt
import pandas as pd

def plot_loss(pickle_file):

    with open(pickle_file, 'rb') as f:
        loss = pickle.load(f)

    loss = pd.DataFrame(loss, columns=['Iter', 'Loss'])
    print(loss)   
    plt.figure(figsize=(10, 6))
    plt.plot(loss['Iter'], loss['Loss'], alpha=0.5, color='blue')
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    # plt.yticks(loss['Loss'].unique())
    # plt.xticks(loss['Iter'].unique())
    plt.legend()
    plt.show()

plot_loss('src/data/loss/loss_best.pkl')
from typing import List
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(name: str,loss_history: List[float], eval_loss_history: List[float], training_x: List[int], eval_x: List[int], epochs: int):
    fig, ax = plt.subplots()
    N = training_x[-1]
    ax.plot(training_x, loss_history, label='Training Loss')
    ax.plot(eval_x, eval_loss_history, label='Evaluation Loss')
    ax.set_xticks([int(i/epochs * N) for i in range(epochs)])
    ax.set_xticklabels([f'Epoch {i}' for i in range(epochs)])

    ax.legend()
    ax.set_title(f'{name} Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Step')
    plt.savefig(f'figures/{name}_loss.png')

def plot_accuracy(name: str, accuracy: List[float], top5Accuracy: List[float], eval_x: List[int], epochs: int):
    fig, ax = plt.subplots()
    N = eval_x[-1]
    ax.plot(eval_x, accuracy, label='Top 1 accuracy')
    ax.plot(eval_x, top5Accuracy, label='Top 5 accuracy')
    ax.set_xticks([int(i/epochs * N) for i in range(epochs)])
    ax.set_xticklabels([f'Epoch {i}' for i in range(epochs)])

    ax.set_title(f'{name} Accuracy')
    ax.legend()
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Step')
    plt.savefig(f'figures/{name}_accuracy')
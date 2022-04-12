import matplotlib.pyplot as plt
import numpy as np

def plot_loss(name: str,loss_history: np.ndarray, epochs: int):
    fig, ax = plt.subplots()
    N = len(loss_history)
    for epoch in range(epochs):
        X = np.arange(int(epoch * N/epochs), int((epoch+1) * N/epochs), dtype=int)
        ax.plot(X, loss_history[X], label=f'Epoch {epoch+1}')
    ax.legend()
    ax.set_title(f'{name} Loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batch')
    plt.savefig(f'figures/{name}_loss.png')
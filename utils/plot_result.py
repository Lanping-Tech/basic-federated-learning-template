import os
from matplotlib import pyplot as plt

def plot_graph(X, y, format = '-', label=''):
    plt.plot(X, y, format, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.grid(True)

def plot_result(metric_collection, n_rounds, results_dir):

    fig = plt.figure(figsize=(10, 6))
    plot_graph(list(range(1, n_rounds+1)), metric_collection['train_acc'], label='Train Accuracy')
    plot_graph(list(range(1, n_rounds+1)), metric_collection['val_loss'], label='Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "federated_model_Accuracy.png"))
    plt.cla()

    plt.figure(figsize=(10, 6))
    plot_graph(list(range(1, n_rounds+1)), metric_collection['train_loss'], label='Train Loss')
    plot_graph(list(range(1, n_rounds+1)), metric_collection['val_loss'], label='Validation loss')
    plt.legend()
    plt.savefig(os.path.join(results_dir, "federated_model_loss.png"))

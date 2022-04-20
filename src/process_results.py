
import json
import os
import sys

from plot import plot_accuracy, plot_loss




def main(results: str, name: str):
    learning_x = []
    eval_x = []
    learning_loss = []
    eval_loss = []
    accuracy = []
    top5Accuracy = []
    for log in results['log_history']:
        if 'loss' in log:
            learning_loss.append(log['loss'])
            learning_x.append(log['step'])
        else:
            eval_x.append(log['step'])
            eval_loss.append(log['eval_loss'])
            accuracy.append(log['eval_accuracy'])
            # The ternary just here for me to debug
            top5Accuracy.append(log['eval_top5_accuracy'] if 'eval_top5_accuracy' in log else log['eval_accuracy'])
    
    plot_loss(name, learning_loss, eval_loss, learning_x, eval_x, 3)
    plot_accuracy(name, accuracy, top5Accuracy, eval_x, 3)

if __name__ == "__main__":
    result_path = sys.argv[1]
    name = sys.argv[2] if sys.argv[2] != '' else os.path.splitext(os.path.basename(result_path))[0]
    with open(result_path) as f:
        results = json.load(f)
    main(results, name)
# Copyright 2021, Lanping-Tech.

"""Runs federated training."""
import os
import argparse

from tensorflow.keras import losses, metrics, optimizers

import tensorflow_federated as tff

from federated_dataset import load_federated_data
from federated_model import get_federated_model_from_keras

from utils.plot_result import plot_result
import nest_asyncio
nest_asyncio.apply()

tff.backends.native.set_local_execution_context(clients_per_thread=5)

if __name__ == "__main__":

    # Parsing arguments and setting hyper-parameters
    parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--experiment_name', default='Federated_Learning', type=str, help='Federated learning experiment name.')
    parser.add_argument('--model_name', default='resnet18', type=str, choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help='Federated learning model name.')
    parser.add_argument('--dataset_path', default='dataset', type=str, help='Federated learning dataset path.')

    parser.add_argument('--n_clients', default=5, type=int, help='Number of clients')
    parser.add_argument('--n_epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--n_rounds', default=5, type=int, help='Number of rounds')
    parser.add_argument('--client_lr', default=1e-3, type=float, help='Client learning rate.')
    parser.add_argument('--server_lr', default=1e-3, type=float, help='Server learning rate.')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size.')
    parser.add_argument('--crop_shape', default=64, type=int, help='Crop size.')

    parser.add_argument('--save_dir', default='models', type=str, help='The path where the model is saved.')
    parser.add_argument('--results_dir', default='results', type=str, help='The path where the result is saved.')

    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)

    input_shape = (args.crop_shape, args.crop_shape, 3)

    federated_train_data, federated_test_data, n_classes = load_federated_data(args.dataset_path, args.n_clients, args.n_epochs, args.batch_size, args.crop_shape)

    federated_model = get_federated_model_from_keras(args.model_name, input_shape, federated_train_data[0].element_spec, n_classes)

    iterative_process = tff.learning.build_federated_averaging_process(
        federated_model,
        client_optimizer_fn=lambda: optimizers.Adam(learning_rate=args.client_lr),
        server_optimizer_fn=lambda: optimizers.Adam(learning_rate=args.server_lr))

    state = iterative_process.initialize()

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    best_eval_model = None
    best_eval_acc = 0.0
    for round_num in range(1, args.n_rounds+1):
        state, tff_metrics = iterative_process.next(state, federated_train_data)

        eval_fn = tff.learning.build_federated_evaluation(federated_model)
        current_model = iterative_process.get_model_weights(state)
        validation_metrics = eval_fn(current_model, federated_test_data) # test data
        
        print('round {:2d}, metrics={}'.format(round_num, tff_metrics))
        print(f"Eval loss : {validation_metrics['loss']} and Eval accuracy : {validation_metrics['categorical_accuracy']}")
        train_acc.append(float(tff_metrics['train']['categorical_accuracy']))
        val_acc.append(validation_metrics['categorical_accuracy'])
        train_loss.append(float(tff_metrics['train']['loss']))
        val_loss.append(validation_metrics['loss'])

    metric_collection = {"train_acc": train_acc,
                         "val_acc": val_acc,
                         "train_loss": train_loss,
                         "val_loss": val_loss}
    
    if best_eval_model is not None:
        best_eval_model.save(os.path.join(args.save_dir, 'model.h5'))

    plot_result(metric_collection, args.n_rounds, args.results_dir)





    

    

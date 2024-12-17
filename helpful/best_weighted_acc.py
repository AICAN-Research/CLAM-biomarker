import os
import subprocess
import time
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
import ast
import pandas as pd
import numpy as np
import tabulate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and Sort Experiments")
    parser.add_argument("--path", type=str, required=True, help="Path to the folder containing experiments")
    parser.add_argument("--min_epochs", type=int, required=True, help="Minimum number of epochs to consider an experiment")
    parser.add_argument("--model_size", type=str, default="", help="Filter experiments by model size")
    return parser.parse_args()

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def parse_metrics(log_dir, weights=(0.8, 0.2)):
    """
    Parse metrics from TensorBoard log files and compute the average weighted accuracy
    across all event files in the log directory. Skips event files where scalars are not found.
    """
    event_files = []
    for dirpath, _, filenames in os.walk(log_dir):
        for filename in filenames:
            if filename.startswith('events.out'):
                event_files.append(os.path.join(dirpath, filename))

    weighted_means = []

    for event_file in event_files:
        try:
            event_accumulator = EventAccumulator(event_file)
            event_accumulator.Reload()

            # Retrieve accuracy scalars
            class_0_acc = event_accumulator.Scalars('final/val_class_0_acc')
            class_1_acc = event_accumulator.Scalars('final/val_class_1_acc')

            # Skip if scalars are missing
            if not class_0_acc or not class_1_acc:
                print(f"Skipping {event_file}: Required scalars not found.")
                continue

            # Assuming you want to use the latest values
            latest_class_0_acc = max(scalar.value for scalar in class_0_acc)
            latest_class_1_acc = max(scalar.value for scalar in class_1_acc)

            # Compute the weighted mean
            weighted_mean = (weights[0] * latest_class_0_acc +
                             weights[1] * latest_class_1_acc)

            weighted_means.append(weighted_mean)  # Append to list
        except Exception as e:
            print(f"Error processing event file {event_file}: {e}")

    # Calculate average weighted accuracy
    return np.mean(weighted_means) if weighted_means else 0



def get_max_epoch_from_subfolders(experiment_path, scalar_name='val/loss'):
    """
    Get the maximum epoch number for a scalar across all subfolders in an experiment directory.
    """
    max_epoch = 0
    try:
        for subfolder in os.listdir(experiment_path):
            subfolder_path = os.path.join(experiment_path, subfolder)
            if os.path.isdir(subfolder_path):
                try:
                    event_accumulator = EventAccumulator(subfolder_path)
                    event_accumulator.Reload()
                    if scalar_name in event_accumulator.Tags().get('scalars', []):
                        scalars = event_accumulator.Scalars(scalar_name)
                        max_epoch = max(max_epoch, max(scalar.step for scalar in scalars))
                except Exception as e:
                    print(f"Error processing subfolder {subfolder_path}: {e}")
    except Exception as e:
        print(f"Error processing experiment path {experiment_path}: {e}")
    return max_epoch

def filter_experiment_by_model_size(experiment_path, model_size):
    """
    Check if the experiment matches the 'model_size' filter.
    """
    params_file = None
    for file in os.listdir(experiment_path):
        if file.startswith("experiment_"):
            params_file = os.path.join(experiment_path, file)
            break

    if params_file and os.path.isfile(params_file):
        try:
            with open(params_file, "r") as file:
                params = ast.literal_eval(file.read())
                return params.get('model_size') == model_size
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing parameters file {params_file}: {e}")
            return False
    return False

def main():
    args = parse_args()

    # Validate the root folder
    if not os.path.isdir(args.path):
        print(f"Error: {args.path} is not a valid directory.\n")
        return

    # List all experiments
    all_experiments = [
        os.path.join(args.path, exp)
        for exp in os.listdir(args.path)
        if os.path.isdir(os.path.join(args.path, exp))
    ]

    experiments = [exp for exp in all_experiments if filter_experiment_by_model_size(exp, args.model_size)]
    print("Filtered for model size.\n")
    print(f"Considered Experiments: {len(experiments)}.\n")
    # Filter experiments by minimum epochs
    experiments = [
        exp for exp in experiments if get_max_epoch_from_subfolders(exp) >= args.min_epochs
    ]
    print("Filtered for epochs.\n")

    # Further filter experiments by model size


    print(f"Considered Experiments: {len(experiments)}.\n")


    if not experiments:
        print("No experiments matching the criteria found in the specified folder.\n")
        return

    # Compute metrics for each experiment
    experiment_results = []
    for experiment in experiments:
        weighted_mean = parse_metrics(experiment)
        experiment_results.append({
            "Experiment": experiment,
            "Average Weighted Mean": weighted_mean
        })

    # Sort by weighted mean accuracy
    experiment_results.sort(key=lambda x: x["Average Weighted Mean"], reverse=True)

    # Display results in a table
    df = pd.DataFrame(experiment_results)
    print("\nSorted Experiment Results:")
    print(df.to_markdown(index=False))

if __name__ == "__main__":
    main()

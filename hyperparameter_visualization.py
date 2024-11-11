from pathlib import Path
import os
import pandas as pd
import optuna
import sqlite3
import numpy as np
from optuna.importance import get_param_importances
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import shutil

def visualize_study(study_id):
    study_name = get_study_name(study_id) # 5,6
    study = optuna.load_study(study_name=study_name, storage="sqlite:///example.db")

    param_importances = get_param_importances(study)

    # Print the importance of each parameter
    for param, importance in param_importances.items():
        print(f"Hyperparameter: {param}, Importance: {importance}")

    print(f"Best trial is  {study.best_trial.number} with value {study.best_value} . Starttime : {study.best_trial.datetime_start}")
    print(f"The parameters are : {study.best_params}")


def get_study_name(study_id):

    conn = sqlite3.connect('example.db')
    cursor = conn.cursor()
    cursor.execute('SELECT study_name FROM studies WHERE study_id = ?', (study_id,))
    study_name = cursor.fetchone()[0]
    conn.close()

    print(f"Study Name: {study_name}")
    return study_name

def parse_metrics(log_dir, weights=(0.8, 0.2)):
    best_weighted_mean = 0
    best_trial = None
    delete = True

    event_files = []
    for dirpath, _, filenames in os.walk(log_dir):
        for filename in filenames:
            if filename.startswith('events.out'):
                event_files.append(os.path.join(dirpath, filename))

    for idx, event_file in enumerate(event_files):
        event_accumulator = EventAccumulator(event_file)
        event_accumulator.Reload()

        # if len(event_accumulator.Scalars('val/loss')) > 102:
        if True:
            delete = False


            # Get the scalar values for each class
            class_0_acc = event_accumulator.Scalars('final/val_class_0_acc')
            class_1_acc = event_accumulator.Scalars('final/val_class_1_acc')

            if class_0_acc and class_1_acc:
                # Assuming you want to use the latest values
                latest_class_0_acc = max(scalar.value for scalar in class_0_acc)
                latest_class_1_acc = max(scalar.value for scalar in class_1_acc)

                # Compute the weighted mean
                weighted_mean = (weights[0] * latest_class_0_acc +
                                 weights[1] * latest_class_1_acc)

                # Update the best weighted mean
                if weighted_mean > best_weighted_mean:
                    best_weighted_mean = weighted_mean
                    best_trial = idx
    # if delete: # (delete all models that have only run for 101 epochs )
    #     shutil.rmtree(log_dir)
    # #     # print(log_dir)

    return best_weighted_mean


def final_scores(root_dir):

    event_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith('events.out'):
                event_files.append(os.path.join(dirpath, filename))

    # Define the metrics of interest
    metrics = [
        'final/test_auc',
        'final/test_class_0_acc',
        'final/test_class_1_acc',
        'final/val_auc',
        'final/val_class_0_acc',
        'final/val_class_1_acc'
    ]

    # Initialize dictionaries to store metrics
    metrics_data = {metric: [] for metric in metrics}

    # Process each event file
    for event_file in event_files:
        metrics_data = extract_final_scores(event_file, metrics, metrics_data)

    # Calculate mean and variance for each metric
    results = {}
    for metric, values in metrics_data.items():
        if values:
            mean = np.mean(values)
            std = np.std(values)
            results[metric] = {'mean': mean, 'standarddev': std}
        else:
            results[metric] = {'mean': None, 'standarddev': None}

    # Print the results
    for metric, result in results.items():
        print(f"{metric}: Mean = {result['mean']:.4f}, Standard Deviation = {result['standarddev']:.4f}")



def extract_final_scores(event_file, metrics, metrics_data):
    event_accumulator = EventAccumulator(event_file)
    event_accumulator.Reload()

    for metric in metrics:
        try:
            # Extract scalar values for the metric
            scalar_values = event_accumulator.scalars.Items(metric)
            values = [item.value for item in scalar_values]
            if values:
                metrics_data[metric].extend(values)
        except KeyError:
            print(f"Metric {metric} not found in {event_file}")
    return metrics_data


if __name__ == "__main__":
    ''' Visualize hyperparameter importance of a study '''

    # study_id = 12 #12,13
    # visualize_study(study_id)

    ''' Show top 10 experiments '''

    root_dir = Path(r'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/results/1024')
    data = []

    for folder in Path(root_dir).iterdir():
        try:
            weighted_mean =parse_metrics(folder,weights=(0.8, 0.2) )
            # print(f'{folder.stem} has weighted mean : {weighted_mean}' )
            data.append({'folder_name': folder.stem, 'weighted_mean': weighted_mean})
        except:
            continue

    df = pd.DataFrame(data)
    top_10 = df.nlargest(30, 'weighted_mean')
    pd.set_option('display.max_rows', None)
    print(top_10)

    ''' Caclculate mean and standard deviation of selected model'''
    # root_dir = '/mnt/EncryptedDisk2/BreastData/Studies/CLAM/results/256/clam_sb220924_082605_s1'
    # final_scores(root_dir)



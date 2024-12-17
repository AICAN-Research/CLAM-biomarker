import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Base directory containing experiments
results_base_dir = '/mnt/EncryptedDisk2/BreastData/Studies/CLAM/results/256/'

# 256
experiment_names = [
    'clam_sb101224_035019_s1',
    'clam_mb261124_065549_s1',
    'clam_sb251124_131731_s1',
    'clam_mb051224_224656_s1',
    'clam_sb281124_004147_s1',
    'clam_mb281124_202623_s1',
    'clam_sb251124_131829_s1',
    'clam_mb271124_120916_s1',
    'clam_sb121224_234654_s1',
    'clam_mb281124_043602_s1',
    'clam_sb101224_010104_s1',
    'clam_mb261124_035124_s1'
]
#1024
# experiment_names = [
# 'clam_sb291124_224838_s1' ,
# 'clam_mb101224_134421_s1',
# 'clam_sb281124_010319_s1',
# 'clam_mb271124_152545_s1',
# 'clam_sb251124_130735_s1',
# 'clam_mb021224_225232_s1',
# 'clam_mb291124_064509_s1',
# 'clam_sb261124_012250_s1',
# 'clam_sb101224_230051_s1',
# 'clam_mb121224_205925_s1',
# 'clam_sb121224_054755_s1' ,
# 'clam_mb261124_171203_s1'
# ]

def extract_val_acc_from_txt(file_path):
    """
    Extract 'val_acc' values from a text file.
    """
    val_acc_values = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "val_acc" in line:
                    # Assuming lines with val_acc look like 'val_acc: 0.85'
                    try:
                        val_acc_values.append(float(line.split(":")[1].strip()))
                    except (IndexError, ValueError):
                        print(f"Malformed line in {file_path}: {line}")
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return val_acc_values

def extract_metrics_from_events(event_dir):
    """
    Extract metrics from all event files in a given directory.
    """
    val_acc_0_values = []
    val_acc_1_values = []
    val_auc_values = []

    for root, _, files in os.walk(event_dir):
        for file in files:
            if file.startswith("events.out"):
                event_file = os.path.join(root, file)
                try:
                    event_accumulator = EventAccumulator(event_file)
                    event_accumulator.Reload()

                    # Extract metrics if they exist
                    if 'final/val_class_0_acc' in event_accumulator.Tags().get('scalars', []):
                        val_acc_0 = event_accumulator.Scalars('final/val_class_0_acc')
                        val_acc_0_values.append(max(scalar.value for scalar in val_acc_0))

                    if 'final/val_class_1_acc' in event_accumulator.Tags().get('scalars', []):
                        val_acc_1 = event_accumulator.Scalars('final/val_class_1_acc')
                        val_acc_1_values.append(max(scalar.value for scalar in val_acc_1))

                    if 'final/val_auc' in event_accumulator.Tags().get('scalars', []):
                        val_auc = event_accumulator.Scalars('final/val_auc')
                        val_auc_values.append(max(scalar.value for scalar in val_auc))
                except Exception as e:
                    print(f"Error processing event file {event_file}: {e}")

    # Return means of each metric if values exist

    return {
        "mean_val_acc_0": f"{np.mean(val_acc_0_values):.4f} ± {np.var(val_acc_0_values):.4f}" if val_acc_0_values else None,
        "mean_val_acc_1": f"{np.mean(val_acc_1_values):.4f} ± {np.var(val_acc_1_values):.4f}" if val_acc_1_values else None,
        "mean_val_auc": f"{np.mean(val_auc_values):.4f} ± {np.var(val_auc_values):.4f}" if val_auc_values else None,
    }


def calculate_averages_for_experiments(base_dir, experiment_names):
    """
    Calculate average metrics for multiple experiments.
    """
    results = []  # To store results for each experiment

    for experiment_name in experiment_names:
        experiment_path = os.path.join(base_dir, experiment_name)
        summary_csv_path = os.path.join(experiment_path, "summary.csv")

        df = pd.read_csv(summary_csv_path)

        # Calculate the averages
        mean_val_acc = df["val_acc"].mean()
        variance_val_acc = df["val_acc"].var()
        formatted_val_acc = f"{mean_val_acc:.4f} ± {variance_val_acc:.4f}"

        # Extract metrics from TensorBoard event files
        event_metrics = extract_metrics_from_events(experiment_path)


        # Combine all metrics
        results.append({
            "experiment_name": experiment_name,
            "mean_val_acc": formatted_val_acc,
            "mean_val_acc_0": event_metrics.get("mean_val_acc_0"),
            "mean_val_acc_1": event_metrics.get("mean_val_acc_1"),
            "mean_val_auc": event_metrics.get("mean_val_auc"),
        })

    # Convert results to a DataFrame for better visualization
    results_df = pd.DataFrame(results)
    return results_df

# Calculate and summarize results
results_df = calculate_averages_for_experiments(results_base_dir, experiment_names)

# Print a summary table
if not results_df.empty:
    print("\nSummary Table:")
    print(results_df.to_markdown(index=False))
else:
    print("No valid experiments processed.")

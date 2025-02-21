import os
import torch
from collections import OrderedDict
from utils.eval_utils import initiate_model
import ast

# Directory containing experiments
results_base_dir = '/.../results/2048'

# List of experiment names (subdirectories in results_base_dir)

experiment_names = []

def read_experiment_params(exp_dir):
    """
    Reads the experiment parameters from the configuration file in the experiment directory.
    """
    experiment_name = os.path.basename(exp_dir).split('_s1')[0]
    config_file = os.path.join(exp_dir, f"experiment_{experiment_name}.txt")
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Experiment configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        params = ast.literal_eval(f.read())  # Convert string to dictionary
    return params


def clean_state_dict(state_dict):
    """
    Cleans a state dictionary by removing unwanted keys and handling `DataParallel` prefixes.
    - Removes keys containing 'instance_loss_fn'.
    - Removes '.module' prefix from key names.
    """
    cleaned_state_dict = OrderedDict()
    for key in state_dict.keys():
        # Skip keys containing 'instance_loss_fn'
        if 'instance_loss_fn' in key:
            continue
        # Remove '.module' from key names and add to cleaned_state_dict
        cleaned_state_dict[key.replace('.module', '')] = state_dict[key]
    return cleaned_state_dict


def average_checkpoints(path_to_checkpoint_dir, args, device):
    """
    Averages the checkpoints in the given directory using the specified model arguments.
    """
    # List checkpoint files
    checkpoint_files = [
        os.path.join(path_to_checkpoint_dir, f"s_{i}_checkpoint.pt") for i in range(10)
    ]

    # Initialize the model using the first checkpoint
    initial_ckpt = checkpoint_files[0]
    model = initiate_model(args, initial_ckpt, device)

    # Load all state dictionaries from checkpoints and clean them
    state_dicts = [clean_state_dict(torch.load(ckpt, map_location=device)) for ckpt in checkpoint_files]

    # Clean and average state dictionaries
    averaged_state_dict = OrderedDict()
    for key in state_dicts[0]:  # Iterate over all parameter keys
        # Ensure all keys are present in all state_dicts
        if all(key in sd for sd in state_dicts):
            # Stack parameters
            stacked_tensors = torch.stack([sd[key] for sd in state_dicts])

            # Convert to float if needed
            if stacked_tensors.dtype not in (torch.float32, torch.float64, torch.complex64, torch.complex128):
                stacked_tensors = stacked_tensors.float()

            # Compute the mean
            averaged_state_dict[key] = stacked_tensors.mean(dim=0)
        else:
            print(f"Key {key} missing in one or more state_dicts, skipping.")

    # Load averaged weights into the model
    model.load_state_dict(averaged_state_dict, strict=False)

    # Save the ensembled model
    averaged_model_path = os.path.join(path_to_checkpoint_dir, "averaged_model.pt")
    torch.save(model.state_dict(), averaged_model_path)
    print(f"Averaged model saved as: {averaged_model_path}")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for experiment_name in experiment_names:
        exp_dir = os.path.join(results_base_dir, experiment_name)

        # Read experiment parameters
        try:
            params = read_experiment_params(exp_dir)
        except FileNotFoundError as e:
            print(e)
            continue

        # Set arguments dynamically from experiment parameters
        class Args:
            drop_out = params.get('use_drop_out', 0.0)
            n_classes = 2  # Example value, adjust as needed
            embed_dim = 1024  # Example value, adjust as needed
            model_size = params.get('model_size', 'big')
            model_type = params.get('model_type', 'clam_mb')

        args = Args()

        # Path to the checkpoint directory for the current experiment
        path_to_checkpoint_dir = os.path.join(exp_dir)

        # Average the checkpoints for the current experiment
        try:
            average_checkpoints(path_to_checkpoint_dir, args, device)
        except Exception as e:
            print(f"Error processing experiment {experiment_name}: {e}")


if __name__ == "__main__":
    main()

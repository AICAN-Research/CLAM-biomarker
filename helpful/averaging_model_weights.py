import torch
from collections import OrderedDict
from utils.eval_utils import initiate_model

# Define the arguments required for initiating the model
class Args:
    drop_out =  0.06813855689777648  # Example value; replace with the correct value
    n_classes = 2   # Example value; replace with the correct value
    embed_dim = 1024  # Example value; replace with the correct value
    model_size = "big"  # Example value; replace with the correct value
    model_type = "clam_mb"  # Example: "clam_sb", "clam_mb", or "mil"

args = Args()

# Directory containing checkpoint files
path_to_checkpoint_dir = '/mnt/EncryptedDisk2/BreastData/Studies/CLAM/results/1024/clam_mb131124_050616_s1'

# List of checkpoint files (ensure they exist in the specified directory)
checkpoint_files = [
    f"{path_to_checkpoint_dir}/s_0_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_1_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_2_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_3_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_4_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_5_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_6_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_7_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_8_checkpoint.pt",
    f"{path_to_checkpoint_dir}/s_9_checkpoint.pt"
]

# Initialize the model using the first checkpoint
initial_ckpt = checkpoint_files[0]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = initiate_model(args, initial_ckpt, device)

# Load all state dictionaries from checkpoints
state_dicts = [torch.load(ckpt, map_location=device) for ckpt in checkpoint_files]

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
model.load_state_dict(averaged_state_dict)

# Save the ensembled model
torch.save(model.state_dict(), f"{path_to_checkpoint_dir}/averaged_model.pt")
print("Averaged model saved as 'averaged_model.pt'")

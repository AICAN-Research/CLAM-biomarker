import pandas as pd

size = '1024'

# models = [
#     "clam_sb101224_035019_s1",
#     "clam_mb261124_065549_s1",
#     "clam_sb251124_131731_s1",
#     "clam_mb051224_224656_s1",
#     "clam_sb281124_004147_s1",
#     "clam_mb281124_202623_s1",
#     "clam_sb251124_131829_s1",
#     "clam_mb271124_120916_s1",
#     "clam_sb121224_234654_s1",
#     "clam_mb281124_043602_s1",
#     "clam_sb101224_010104_s1",
#     "clam_mb261124_035124_s1", ]


models = [
    "clam_sb291124_224838_s1",
    "clam_mb101224_134421_s1",
    "clam_sb281124_010319_s1",
    "clam_mb271124_152545_s1",
    "clam_sb251124_130735_s1",
    "clam_mb021224_225232_s1",
    "clam_sb261124_012250_s1",
    "clam_mb291124_064509_s1",
    "clam_sb101224_230051_s1",
    "clam_mb121224_205925_s1",
    "clam_sb121224_054755_s1",
    "clam_mb261124_171203_s1"
]

# Loop through each model
results = []
for model in models:
    # Replace size and model in the path
    file_path = f'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/eval/{size}/{model}/fold_0.csv'

    try:
        # Read the CSV file for the current model
        data = pd.read_csv(file_path)

        # Count initial entries per class
        initial_counts = data['Y'].value_counts()

        # Calculate initial class-wise accuracies before filtering
        initial_accuracies = data.groupby('Y').apply(
            lambda group: (group['Y'] == group['Y_hat']).mean()
        )

        # Filter rows explicitly using .loc
        filtered_data = data.loc[
            ~((data['p_0'].between(0.4, 0.6))  (data['p_1'].between(0.4, 0.6)))
        ]

        # Count remaining entries per class
        remaining_counts = filtered_data['Y'].value_counts()

        # Calculate class-wise accuracies after filtering
        filtered_accuracies = filtered_data.groupby('Y').apply(
            lambda group: (group['Y'] == group['Y_hat']).mean()
        )

        # Calculate how many lines were removed
        removed_counts = initial_counts - remaining_counts

        # Store results
        results.append({
            "model": model,
            "initial_accuracies": initial_accuracies.to_dict(),
            "filtered_accuracies": filtered_accuracies.to_dict(),
            "removed_counts": removed_counts.to_dict()
        })

    except Exception as e:
        print(f"Error processing model {model}: {e}")

    # Print results for each model
for result in results:
    print(f"Model: {result['model']}")

    print("\nClass-wise accuracies (before filtering):")
    for class_label, accuracy in result["initial_accuracies"].items():
        print(f"  Class {class_label}: {accuracy * 100:.2f}%")

    print("\nClass-wise accuracies (after filtering):")
    for class_label, accuracy in result["filtered_accuracies"].items():
        print(f"  Class {class_label}: {accuracy * 100:.2f}%")

    print("\nDifference between initial and filtered accuracies:")
    for class_label in result["initial_accuracies"]:
        initial_acc = result["initial_accuracies"].get(class_label, 0)
        filtered_acc = result["filtered_accuracies"].get(class_label, 0)
        diff = filtered_acc - initial_acc
        print(f"  Class {class_label}: {diff * 100:.2f}%")

    print("\nNumber of lines removed for each class:")
    for class_label, removed in result["removed_counts"].items():
        print(f"  Class {class_label}: {removed} lines removed")

    print("-" * 50)
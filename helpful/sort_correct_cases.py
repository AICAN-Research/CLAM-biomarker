import pandas as pd
size = '1024'
model ='clam_mb121224_205925_s1'


file_path = f'/mnt/EncryptedDisk2/BreastData/Studies/CLAM/eval/{size}/{model}/fold_0.csv'

data = pd.read_csv(file_path)

filtered_data = data[~((data['p_0'].between(0.0, 0.95)) & (data['p_1'].between(0.000, 0.9995)))]

correctly_classified = filtered_data[filtered_data['Y'] == filtered_data['Y_hat']]
correctly_classified_sorted = correctly_classified.sort_values(by='Y')

print(f"Model: {model}")
for class_label in correctly_classified_sorted['Y'].unique():
    print(f"\nClass {class_label}:")

    # Filter data for the current class
    class_data = correctly_classified_sorted[correctly_classified_sorted['Y'] == class_label]

    # Sort by p_0 for class 0, and by p_1 for class 1
    if class_label == 0:
        class_data = class_data.sort_values(by='p_0', ascending=False)
    elif class_label == 1:
        class_data = class_data.sort_values(by='p_1', ascending=False)

    # Print the slide details
    for _, row in class_data.iterrows():
        print(f"  Slide ID: {int(row['slide_id'])}, p_0: {row['p_0']:.4f}, p_1: {row['p_1']:.4f}")
print("-" * 50)

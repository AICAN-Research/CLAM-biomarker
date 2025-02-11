import pandas as pd
import pickle
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Paths to datasets
testset_1 = '/mnt/EncryptedDisk2/BreastData/Studies/CLAM/eval/1024_features/clam_mb121224_205925_s1_testset_1'
HUS = '/mnt/EncryptedDisk2/BreastData/Studies/CLAM/eval/1024_features/clam_mb121224_205925_s1_HUS'
complete = '/mnt/EncryptedDisk2/BreastData/Studies/CLAM/eval/1024_features/clam_mb121224_205925_s1_Testset_1_all'

subtyp_HUS = '/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/CSV/HUS_Maren_P2_few_variables.csv'
subtype_testset1 = '/mnt/EncryptedDisk2/BreastData/Studies/Biomarkers/CSV/biomarkers_291024.csv'

# Load PKL files for Dataset 1 and 2
with open(testset_1 + '/feature_results.pkl', 'rb') as f:
    patient_results_set1 = pickle.load(f)

with open(HUS + '/feature_results.pkl', 'rb') as f:
    patient_results_HUS = pickle.load(f)

with open(complete + '/feature_results.pkl', 'rb') as f:
    patient_results_complete = pickle.load(f)

# Load CSV files for Dataset 1 and 2
csv_set1 = pd.read_csv(testset_1 + '/fold_0.csv')
csv_HUS = pd.read_csv(HUS + '/fold_0.csv')
csv_complete = pd.read_csv(complete + '/fold_0.csv')

csv_subtype_HUS = pd.read_csv(subtyp_HUS)
csv_subtype_testset1 =pd.read_csv(subtype_testset1)

csv_subtype_HUS['slide_id'] = csv_subtype_HUS['kasusnr'] + 6000
csv_subtype_testset1['slide_id'] = csv_subtype_testset1['ID_deltaker']

subtype_mapping = {
    'duct': 'Ductal',
    'lob': 'Lobular',
    'medul': 'Medullary',
    'mucin': 'Mucinous',
    'papil': 'Papillary',
    'metapl': 'Metaplastic',
    'tub': 'Tubular',
    'other': 'Other',

    'duktalt karsinom 85003': 'Ductal',
    'lobulært karsinom 85203': 'Lobular',
    'medullært karsinom 85103': 'Medullary',
    'mukøst karsinom 84803': 'Mucinous',
    'tubulært karsinom 82113': 'Tubular'
}
def match_slide_id_and_label(patient_results, csv_data, subtype_csv, dataset_label):
    data = []
    for slide_id, info in patient_results.items():
        # Find label and prediction
        label_row = csv_data.loc[csv_data['slide_id'] == slide_id]
        if label_row.empty:
            continue  # Skip if slide_id not found

        label = label_row['Y'].values[0]
        prediction = label_row['Y_hat'].values[0]

        # Find subtype and grade
        subtype_row = subtype_csv.loc[subtype_csv['slide_id'] == slide_id]
        if subtype_row.empty:
            subtype, grade = 'Other', None  # Default values if not found
        else:
            if 'TYPE' in subtype_row.columns:  # Testset 1
                raw_subtype = subtype_row['TYPE'].values[0]
                grade = subtype_row['GRAD'].values[0]
            else:  # HUS dataset
                raw_subtype = subtype_row['Diagnose'].values[0]
                grade = subtype_row['histgr'].values[0]

            # Standardize subtype names
            subtype = subtype_mapping.get(raw_subtype, 'Other')
        if grade == 999: continue

        data.append({
            'slide_id': slide_id,
            'label': label,
            'prediction': prediction,
            'subtype': subtype,
            'grade': grade,
            'features': info['features'],
            'dataset': dataset_label
        })

    return pd.DataFrame(data)
# Match data for both datasets
df1 = match_slide_id_and_label(patient_results_set1, csv_set1, csv_subtype_testset1, 'Internal test set')
df2 = match_slide_id_and_label(patient_results_HUS, csv_HUS, csv_subtype_HUS, 'External test set')
df3 = match_slide_id_and_label(patient_results_complete, csv_complete, csv_subtype_testset1, 'Train and validation set')
df3_exclusive = df3[~df3['slide_id'].isin(df1['slide_id'])].copy()



# Separate features for PCA
features_set1 = np.array([np.array(f).flatten() for f in df1['features'].to_list()])
features_set2 = np.array([np.array(f).flatten() for f in df2['features'].to_list()])
features_set3 = np.array([np.array(f).flatten() for f in df3_exclusive['features'].to_list()])

all_features = np.vstack([features_set1, features_set2, features_set3])

# Perform PCA on Dataset 1
pca = PCA(n_components=2)
# reduced_features_set3 = pca.fit_transform(features_set3)
transformation = pca.fit_transform(features_set3)
reduced_features_set3 = pca.transform(features_set3)

# Apply the same transformation to Dataset 2
reduced_features_set1 = pca.transform(features_set1)
reduced_features_set2 = pca.transform(features_set2)

# Add PCA results to DataFrames
df3_exclusive['PCA1'], df3_exclusive['PCA2'] = reduced_features_set3[:, 0], reduced_features_set3[:, 1]
df2['PCA1'], df2['PCA2'] = reduced_features_set2[:, 0], reduced_features_set2[:, 1]
df1['PCA1'], df1['PCA2'] = reduced_features_set1[:, 0], reduced_features_set1[:, 1]

# Combine the DataFrames again for plotting
combined_df = pd.concat([df1, df2], ignore_index=True)


def plot_pca(df, title):
    plt.figure(figsize=(12, 8))

    for _, row in df.iterrows():
        # Determine color based on label
        color = 'red' if row['label'] == 1 else 'blue'

        # Determine marker shape based on dataset
        marker = 'o' if row['dataset'] == 'Internal test set' else '^'

        # Determine edge color based on prediction correctness
        edge_color = 'green' if row['label'] == row['prediction'] else 'black'

        # Plot each point
        plt.scatter(row['PCA1'], row['PCA2'], c=color, marker=marker, edgecolor=edge_color, s=100, linewidth=1.5,
                    label=None)

    # Add legend
    plt.scatter([], [], c='red', marker='s', label='Positive class (y=1)'),  # Red for y=1
    plt.scatter([], [], c='blue', marker='s', label='Negative class (y=0)'),  # Blue for y=0
    plt.scatter([], [], c='white', marker='s', edgecolor='green', label='Correct prediction'),  # Green edge for correct
    plt.scatter([], [], c='white', marker='s', edgecolor='black',
                label='Incorrect prediction'),  # Black edge for incorrect
    plt.scatter([], [], c='black', marker='o', label='Internal test set '),  # Circle for Testset 1
    plt.scatter([], [], c='black', marker='^', label='External test set ')  # Triangle for HUS

    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pca_train(df, title):
    plt.figure(figsize=(12, 8))

    for _, row in df.iterrows():
        # Determine color based on label
        color = 'red' if row['label'] == 1 else 'blue'
        marker = 'o'


        # Plot each point
        plt.scatter(row['PCA1'], row['PCA2'], c=color, marker=marker,edgecolor='black', s=100, linewidth=1.5,
                    label=None)

    # Add legend
    plt.scatter([], [], c='red', marker='o',edgecolor='black', label='y=1')
    plt.scatter([], [], c='blue', marker='o',edgecolor='black',  label='y=0')

    plt.legend(loc='upper left')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_pca_by_subtype(df, title="PCA Projection by Subtype"):
    plt.figure(figsize=(12, 8))

    marker_map = {'Internal test set': 'o', 'External test set': '^', 'Train and validation set': 'o'}  # Define markers per dataset

    # Get unique subtypes and set a consistent color palette
    palette = sns.color_palette("Set1", n_colors=df['subtype'].nunique())

    # Loop through datasets and plot each subset separately with its marker
    for dataset, marker in marker_map.items():
        subset = df[df['dataset'] == dataset]  # Filter dataset-specific rows

        sns.scatterplot(
            x='PCA1', y='PCA2', hue='subtype', palette=palette, data=subset,
            alpha=0.7, s=100, edgecolor='black', legend="full", marker=marker
        )

    subtype_legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=subtype)
        for subtype, color in zip(df['subtype'].unique(), palette)
    ]

    # Create legend for datasets (black markers with different shapes)
    present_datasets = df['dataset'].unique()

    # Create dataset legend only for datasets that exist in df
    dataset_legend_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', markersize=10,linestyle='None', label=dataset)
        for dataset, marker in marker_map.items() if dataset in present_datasets
    ]

    # Combine both legends
    combined_legend = subtype_legend_handles + dataset_legend_handles
    plt.legend(handles=combined_legend, title="Legend", loc='upper left')

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_pca_by_grade(df, title="PCA Projection by Grade"):
    plt.figure(figsize=(12, 8))

    marker_map = {'Internal test set': 'o', 'External test set': '^', 'Train and validation set': 'o'}  # Define markers per dataset

    # Get unique subtypes and set a consistent color palette
    palette = sns.color_palette("Set1", n_colors=df['grade'].nunique())

    # Loop through datasets and plot each subset separately with its marker
    for dataset, marker in marker_map.items():
        subset = df[df['dataset'] == dataset]  # Filter dataset-specific rows

        sns.scatterplot(
            x='PCA1', y='PCA2', hue='grade', palette=palette, data=subset,
            alpha=0.7, s=100, edgecolor='black', legend="full", marker=marker
        )

    grade_legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=grade)
        for grade, color in zip(df['grade'].unique(), palette)
    ]

    # Create legend for datasets (black markers with different shapes)
    present_datasets = df['dataset'].unique()

    # Create dataset legend only for datasets that exist in df
    dataset_legend_handles = [
        plt.Line2D([0], [0], marker=marker, color='black', markersize=10,linestyle='None', label=dataset)
        for dataset, marker in marker_map.items() if dataset in present_datasets
    ]

    # Combine both legends
    combined_legend = grade_legend_handles + dataset_legend_handles
    plt.legend(handles=combined_legend, title="Legend", loc='upper left')

    plt.title(title)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



plot_pca_train(df3_exclusive, 'PCA of Feature Vectors Train and Validation set')
plot_pca(combined_df, 'PCA of Feature Vectors Internal and External test set')
plot_pca_by_subtype(df3_exclusive, 'PCA of Feature Vectors Train and Validation set')
plot_pca_by_subtype(combined_df, 'PCA of Feature Vectors internal and external test set')
plot_pca_by_grade(df3_exclusive, 'PCA of Feature Vectors Train and Validation set')
plot_pca_by_grade(combined_df, 'PCA of Feature Vectors internal and external test set')


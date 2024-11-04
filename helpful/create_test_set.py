"""
Create test set manually from dataset
Needs to be removed from the .csv file later used for create_splits_seg.py by CLAM
Picks equally from pos/neg samples
Based on create_splits_seq.py and CLAM's methods
"""

import os
import pandas as pd
import argparse
import numpy as np


def generate_split(patient_idx, test_num, samples, seed=7):
    """
    Based on generate_split in CLAM's utils.py
    """
    # number of samples (neg/pos) in total in dataset
    indices = np.arange(samples).astype(int)
    # set seed
    np.random.seed(seed)
    # create lists
    all_test_idxs = []
    all_remaining_idxs = []
    # test num can f.ex be [35 15], showing 35 neg and 15 pos samples to be picked from the dataset into the test set
    for c in range(len(test_num)):  # if test_num = [35 15], then len(test_num) = 2
        possible_indices = np.intersect1d(patient_idx[c], indices)  # all indices of this class, intersection of class indices and all indices
        test_idxs = np.random.choice(possible_indices, test_num[c], replace=False)  # choose test ids randomly from possible indices

        remaining_idxs = np.setdiff1d(possible_indices, test_idxs)  # indices of this class left after test

        all_test_idxs.extend(test_idxs)
        all_remaining_idxs.extend(remaining_idxs)

    return all_remaining_idxs, all_test_idxs


parser = argparse.ArgumentParser(description='Creating test set, and remove it from the .csv file later used '
                                             'for create_splits_seq.py')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--shuffle', type=int, default=0,
                    help='shuffle True (1) or False (0). (Default: 0)')
parser.add_argument('--num_classes', type=int, default=2,
                    help='Number of classes. Default 2.')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--strat', type=bool, default= True,
                    help='Stratification, default=True')
args = parser.parse_args()

csv_path='/path/to/file/to/pick/from/.csv'
split_dir = '/path/to/save/dir/'
os.makedirs(split_dir, exist_ok=True)

# read csv-file, drop unneeded columns and reset index
csv_file = pd.read_csv(csv_path)
df = csv_file.drop(columns=['PR', 'HER2', 'Ki67', 'histological subtype', 'histological grade'])
df.reset_index(drop=True, inplace=True)  #@TODO: necessary?

# store ids corresponding each class at the patient or case level
# Divide pos/neg samples on patient/slide leve (same for us)
# list of two arrays, one with index of all 0s and one with all 1s
patient_idxs = [[] for i in range(args.num_classes)]
for i in range(args.num_classes):
    patient_idxs[i] = np.where(df['ER'] == i)[0]

# get nbr of neg/pos in dataset, ex [350 150] = 350 neg and 150 pos samples in dataset
num_slides_cls = np.array([len(cls_ids) for cls_ids in patient_idxs])
# determine number of neg/pos in test set, ex [35 15] = 35 neg, 15 pos
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

# get number of samples in dataset in total (pos + neg)
samples = len(patient_idxs[0]) + len(patient_idxs[1])

# generate split, yields all_remaining_idxs, all_test_idxs
all_remaining_idxs, all_test_idxs = generate_split(patient_idxs, test_num, samples, seed=args.seed)

# match all test idxs with patient ids to make list with two columns: case and id.
slide_ids = [[] for i in range(len(all_test_idxs))]  # list of list of length "samples"
new_df = pd.DataFrame(columns=['case_id', 'slide_id', 'ER'])
for idx in range(len(slide_ids)):
    idx_ = all_test_idxs[idx]
    slide_id = df['slide_id'][idx_]  # slide_id = case_id
    label = df['ER'][idx_]

    new_df.loc[len(new_df)] = [slide_id, slide_id, label]  # slide_id = case_id

new_df.to_csv('/path/to/save/test_set.csv', index=False)
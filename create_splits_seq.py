import pdb
import os
import pandas as pd
from dataset_modules.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping','biomarker_ER', 'biomarker_ER_HUS_1024', 'biomarker_ER_HUS_256','biomarker_ER_2048'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')

args = parser.parse_args()

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=2
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=3
    dataset = Generic_WSI_Classification_Dataset(csv_path = 'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = {'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

elif args.task == 'biomarker_ER':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_256/train_256.csv',
                                  data_dir='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_256/features',
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_col='ER',
                                  label_dict={1: 1, 0: 0}, #label_dict={'>= 1%': 1, '< 1%': 0},
                                  patient_strat=False,
                                  ignore=['Ki67','HER2','PR','histological subtype','histological grade'])
elif args.task == 'biomarker_ER_HUS_1024':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_1024_HUS/train_1024_HUS.csv',
                                  data_dir='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_1024_HUS/features',
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_col='ER',
                                  label_dict={1: 1, 0: 0},  # label_dict={'>= 1%': 1, '< 1%': 0},
                                  patient_strat=False,
                                  ignore=['Ki67', 'HER2', 'PR', 'histological subtype', 'histological grade'])
elif args.task == 'biomarker_ER_2048':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_2048/train_2048.csv',
                                  data_dir='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_2048/features',
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_col='ER',
                                  label_dict={1: 1, 0: 0},  # label_dict={'>= 1%': 1, '< 1%': 0},
                                  patient_strat=False,
                                  ignore=['Ki67', 'HER2', 'PR', 'histological subtype', 'histological grade'])
elif args.task == 'biomarker_ER_HUS_256':
    args.n_classes = 2
    dataset = Generic_MIL_Dataset(csv_path='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_256_HUS/train_256_HUS.csv',
                                  data_dir='/mnt/EncryptedDisk2/BreastData/Studies/CLAM/patchsize_256_HUS/features',
                                  shuffle=False,
                                  seed=args.seed,
                                  print_info=True,
                                  label_col='ER',
                                  label_dict={1: 1, 0: 0},  # label_dict={'>= 1%': 1, '< 1%': 0},
                                  patient_strat=False,
                                  ignore=['Ki67', 'HER2', 'PR', 'histological subtype', 'histological grade'])


else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]


    ''' get ids through first run in debug mode and copy self.test_ids after print(str(self.test_ids)) in test_split_gen '''
    test_ids = [497, 634, 875, 1117, 1035, 387, 1663, 1003, 1206, 847, 1400, 291, 1077, 1431, 592, 157, 1472, 1562, 159, 1495, 1542, 1548, 328, 775, 1172, 1583, 486,
                1481, 128, 1145, 641, 706, 1363, 614, 1304, 504, 125, 1148, 914, 218, 1249, 416, 424, 960, 1555, 1093, 236, 405, 1519, 1361, 1123, 336, 897, 1053, 146,
                221, 818, 827, 671, 1161, 1504, 1073, 1339, 187, 1425, 920, 816, 355, 589, 226, 1695, 1221, 913, 13, 563, 381, 101, 724, 1632, 185, 1505, 1193, 1560,
                254, 740, 1687, 111, 1273, 1594, 1438, 431, 590, 457, 325, 1, 737, 463, 845, 825, 120, 833, 1136, 1518, 1597, 649, 513, 534, 204, 295, 1296, 1344, 465,
                1143, 1698, 122, 206, 390, 1429, 7, 722, 388, 954, 1417, 1507, 1654, 251, 129, 370, 536, 107, 1506, 118, 883, 232, 1096, 795, 1215, 63, 955, 586, 1702,
                766, 482, 1090, 929, 743, 323, 1595, 1664, 811, 1754, 1245, 395, 788, 1163, 468, 1491, 50, 234, 1094, 646, 765, 1376, 340, 1630, 1665, 235, 1600, 1369,
                1072, 377, 1006, 1121, 1207, 1233, 931]


    for lf in label_fracs:
        split_dir = 'splits/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf, custom_test_ids=test_ids) # custom_test_ids=test_ids
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))




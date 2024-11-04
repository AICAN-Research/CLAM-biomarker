"""
Script to create .csv file for training including all cases with corresponding CLAM features for respective patch size
Remove ids which desired biomarker status is not present.
"""

import pandas as pd
from pathlib import Path

biomarker_file = Path(r'/...path/.../...csv')
data_dir = Path(r'/.../path/.../patchsize_X/features/h5_files/')
new_csv = pd.DataFrame(columns=['case_id', 'slide_id', 'Ki67', 'ER','HER2','PR',
                                'histological subtype', 'histological grade'])

biomarker_df = pd.read_csv(biomarker_file)
no_information = []

for file in data_dir.iterdir():
    case_id = file.stem
    slide_id = file.stem

    try:
        biomarker_info = biomarker_df[biomarker_df['ID_deltaker']== int(case_id)]

        new_csv.loc[len(new_csv)] = [case_id, slide_id, biomarker_info['Ki67'].iloc[0],
                                     biomarker_info['ER_1_prosent'].iloc[0] ,biomarker_info['HER2'].iloc[0],
                                     biomarker_info['PR'].iloc[0], biomarker_info['TYPE'].iloc[0],
                                     biomarker_info['GRAD'].iloc[0]]
    except:
        no_information.append(case_id)

# drop if missing desired biomarker status:
new_csv = new_csv.loc[new_csv['ER'].isin([0, 1])]
print(len(new_csv))

new_csv.to_csv('/...path/.../patchsize_X/train_X.csv', index=False)
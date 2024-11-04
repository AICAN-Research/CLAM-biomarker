"""
Match two .csv files to see if they are identical
"""

import pandas as pd
from pathlib import Path


csv_file1 = Path(r'/path/to/file/train_1024.csv')
csv_file2 = Path(r'/path/to/file/train_256.csv')

df1 = pd.read_csv(csv_file1)
df2 = pd.read_csv(csv_file2)

print(df1.compare(df2))



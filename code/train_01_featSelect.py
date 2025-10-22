import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import os
import pickle

df = pd.read_csv('./data/data_withFeat.csv')
df.shape
# (3912, 621)

# =================================================
# Feature selection
# =================================================
df_train = df[df['set'] == 'train']

# Keep only numeric columns
df_num = df_train.select_dtypes(include=[np.number])
df_num = df_num.drop(columns=['class'])

# Remove binary columns with only 5 values
binary_cols = df_num.columns[df_num.nunique() == 2]
remove1 = []
for col in binary_cols:
    min_count = df_num[col].value_counts().min()
    if min_count <= 5:
        remove1.append(col)

df_num = df_num.drop(columns=remove1)

# Remove near zeroVar columns
selector = VarianceThreshold(threshold=0)
selector.fit(df_num)
remove2 = df_num.columns[~selector.get_support()]
df_num = df_num.drop(columns=remove2)

# Compute correlation matrix and remove highly correlated columns
thres = 0.8
cor_matrix = df_num.corr(method='spearman')
upper = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
remove3 = [column for column in upper.columns if any(upper[column].abs() > thres)]
df_num = df_num.drop(columns=remove3)

# Save
keep = ['NP_id', 'class', 'set'] + list(df_num.columns) + ["del_seq", "up5seq", "down5seq"]
df_keep = df[keep]
df_keep.to_csv('./data/data_filteredFeat.csv', index=False)

print(f"Dimensions of filtered dataset: {df_keep.shape}")  
# (3912, 200)
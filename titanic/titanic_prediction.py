import numpy as np
import pandas as pd

train_df = pd.read_csv('./datasets/train.csv')
test_df = pd.read_csv('./datasets/test.csv')
df = [train_df, test_df]

'''
Preview the data
'''
# Display columns
print('Train columns:', train_df.columns.values.tolist())
print('Test columns:', test_df.columns.values.tolist())
print(100 * '-')
# Preview
print(train_df.head())

'''
Preprocessing
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame({
    'a': np.arange(0, 11),
    'b': True
})

# print(df.head())
# print(df.tail())
# print(df.dtypes)
# print(df.values)
# print(df.describe())
# print(df.sort_index(axis=0, ascending=False))
print(df.sort_values(by='a', ascending=False))
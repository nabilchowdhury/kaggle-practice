import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
The data structures

s = pd.Series(data, index=index)
data can be a Python dict, ndarray, or scalar value
index is a list of axis labels
'''
s1 = pd.Series(['data1', 'data2', 'data3', 'data4', 'data5'], index=[1,2,3,4,5])
s1 = pd.Series(np.random.randn(5), index=['a','b','c','d','e'])
s1 = pd.Series(np.random.randn(5))
d = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
s1 = pd.Series(d)
s = pd.Series(d, index=['a', 'b', 'c', 'd', 'e'], name='myseries')
s1 = pd.Series(5, index=['i', 'ii', 'iii', 'iv'])

# print(s.name)
# print(s[:3])
# print(s[:3])
# print(s[s > s.median()])
# print(s[[4, 3, 1]])
# print(np.exp(s))
s['e'] = 4.4
# print('e' in s)
# print(s.get('f'), s.get('f', np.nan)) # Doing s['f'] throws an exception
# print(s * 2)
# print(s[:-1])
# print(s[1:] + s[:-1]) # pd.Series auto align data, unlike numpy

'''
DataFrames
A 2D spreadsheet/SQL table like data structure.
Accepts dict of 1D ndarrays, lists, dicts or Series
        2-D numpy ndarray
        Structured or record array
        A Series
        Another DataFrame
        
Optionally pass index and columns as lists
'''

# From dict of series or dicts
df1 = pd.DataFrame({
    'one': pd.Series([1,2,3], index=['a','b','c']),
    'two': pd.Series([1,2,3,4], index=['a','b','c','d'])
})
df2 = pd.DataFrame(df1, index=['d', 'b', 'a'])
df3 = pd.DataFrame(df1, index=['d', 'b', 'a'], columns=['two', 'three'])
# print(df3)
# print(df.index)
# print(df.columns)

# From dict of lists or ndarrays
dict_ = {'one': [1,2,3,4], 'two': [4,3,2,1]}
df = pd.DataFrame(dict_)
df2 = pd.DataFrame(dict_, index=['a', 'b', 'c', 'd'])
# print(df2)

# From list of dicts
list_of_dicts_ = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 10}]
df = pd.DataFrame(list_of_dicts_)
df2 = pd.DataFrame(list_of_dicts_, index=['first', 'second'])
df3 = pd.DataFrame(list_of_dicts_, columns=['a', 'b'])
# print(df3)

# From a dict of tuples (probably won't ever use this)
df = pd.DataFrame({('a', 'b'): {('A', 'B'): 1, ('A', 'C'): 2},
                ('a', 'a'): {('A', 'C'): 3, ('A', 'B'): 4},
                ('a', 'c'): {('A', 'B'): 5, ('A', 'C'): 6},
                ('b', 'a'): {('A', 'C'): 7, ('A', 'B'): 8},
                ('b', 'b'): {('A', 'D'): 9, ('A', 'B'): 10}})
# print(df)

'''
Column selection, addition, deletion
'''
# Add columns
df1['three'] = df1['one'] * df1['two']
df1['flag'] = df1['one'] > 2

# Delete columns
del df1['two']
three = df1.pop('three')
# print(df1)

# Add scalar
df1['foo'] = 'bar'
# print(df1)

df1['one_trunc'] = df1['one'][:2]
# print(df1)

df1.insert(1, 'bar', df1['one'])
# print(df1)
# print('-' * 100)
'''
Indexing and Selection
Operation                       Syntax          Result

Select col                      df[col]         Series
Select row by label             df.loc[label]   Series
Select row by int location      df.iloc[label]  Series
Slice rows                      df[5:10]        DF
Select rows by bool vector      df[bool_vec]    DF
'''

# print(df1.loc['b'])
# print(df1.iloc[2])

# Data alignment
df_a = pd.DataFrame(np.random.randn(10, 4), columns=['A', 'B', 'C', 'D'])
df_b = pd.DataFrame(np.random.randn(7, 3), columns=['A', 'B', 'C'])
# print(df_a + df_b)
# print(df_a - df_a.iloc[0])

index = pd.date_range('1/1/2000', periods=8)
df = pd.DataFrame(np.random.randn(8, 3), index=index, columns=list('ABC'))
# print(df.sub(df['A'], axis=0))
# print(1 / df)
# print(df * 5 + 1)

# Boolean Operators
df1 = pd.DataFrame({
    'a': [1, 0, 1],
    'b': [1, 1, 0]
}, dtype=bool)

df2 = pd.DataFrame({
    'a': [0, 0, 1],
    'b': [1, 0, 0]
}, dtype=bool)

# print(df1 & df2)
# print(df1.T) # Transposed
# print(df1.info())

# Access columns by doing df.colname
print(df1.b)
'''
OK to compute....	    Nominal	Ordinal	Interval Ratio
frequency distribution.	Yes	    Yes	    Yes	     Yes
median and percentiles.	No	    Yes	    Yes	     Yes
add or subtract.	    No	    No	    Yes	     Yes
mean, standard deviation, standard error of the mean.	No	No	Yes	Yes
ratio, or coefficient of variation.	No	No	No	Yes
 	 	 	 	 
'''

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier

df = {
    'train_df': pd.read_csv('./datasets/train.csv'),
    'test_df': pd.read_csv('./datasets/test.csv')
}

def preview():
    '''
    Preview the data
    '''
    # Display columns
    print('Train columns:', df['train_df'].columns.values.tolist())
    print('Test columns:', df['test_df'].columns.values.tolist())
    print(100 * '-')
    # Preview
    print(df['train_df'].head())
    print(df['train_df'].describe())

    # Females more likely to survive than males
    print(df['train_df'][['Sex', 'Survived']].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False))
    # Class 1 most likely to survive, followed by 2 and 3
    print(df['train_df'][['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False))
    # 1 > 2 > 0 > 3
    print(df['train_df'][['SibSp', 'Survived']].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False))
    # 3 > 1 > 2 > 0
    print(df['train_df'][['Parch', 'Survived']].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False))
    # C > Q > S
    print(df['train_df'][['Embarked', 'Survived']].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False))
    # Drop cabin
    # print(train_df[['Cabin', 'Survived']].groupby(['Cabin']).mean().sort_values(by='Survived', ascending=False))

def visualize():
    '''
    Visualize Age, Ticket, Fare
    '''
    # Certain age bands more likely to survive
    g_age = sns.FacetGrid(df['train_df'], col='Survived')
    g_age.map(plt.hist, 'Age', bins=20)
    plt.show()

    grid = sns.FacetGrid(df['train_df'], col='Survived')
    grid.map(plt.hist, 'Fare', bins=20)
    plt.show()

    grid = sns.FacetGrid(df['train_df'], row='Embarked', col='Survived', size=2.2, aspect=1.6)
    grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
    grid.add_legend()
    plt.show()

def check_title():
    '''
    Extract Title from Name
    '''
    # Confirmed Title as an useful feature, let's keep it (after some processing)
    train_df2 = df['train_df'].copy()
    train_df2['Title'] = train_df2['Name'].str.extract('([A-Za-z]+\.)', expand=False)
    print(train_df2[['Title', 'Survived']].groupby(['Title']).agg(['mean', 'count']).sort_values(by=[('Survived', 'mean')], ascending=False))

'''
Processing
'''
# Do one-hot encoding for all categorical features (using dummy vars)
# Maybe combine SibSp and Parch into single feature?

for key in df.keys():
    df[key]['Title'] = df[key]['Name'].str.extract('([A-Za-z]+\.)', expand=False)
    df[key].drop(['Name'], axis=1, inplace=True)

for key in df.keys():
    df[key] = pd.get_dummies(df[key], columns=['Sex', 'Embarked', 'Pclass'])
    df[key].drop(['Cabin', 'Ticket', 'PassengerId', 'SibSp', 'Parch', 'Age', 'Fare', 'Title'], axis=1, inplace=True) # Also drop Cabin as we won't use it


[print(data.head(), '\n', '-' * 100) for data in df.values()]
# [data.fillna(0) for data in df.values()]
'''
Prediction
'''
X_train = df['train_df'].drop('Survived', axis=1)
Y_train = df['train_df']['Survived']
X_test = df['test_df'].copy()

logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)
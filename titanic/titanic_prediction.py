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
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

df = {
    'train_df': pd.read_csv('./datasets/train.csv'),
    'test_df': pd.read_csv('./datasets/test.csv')
}

passengerIds = df['test_df']['PassengerId']

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


def check_title(processed=False):
    '''
    Extract Title from Name
    '''
    # Confirmed Title as an useful feature, let's keep it (after some processing)
    if processed:
        print(df['train_df'][['Title', 'Survived']].groupby(['Title']).agg(['mean', 'count']).sort_values(
            by=[('Survived', 'mean')], ascending=False))
        return
    train_df2 = df['train_df'].copy()
    train_df2['Title'] = train_df2['Name'].str.extract('([A-Za-z]+\.)', expand=False)
    print(train_df2[['Title', 'Survived']].groupby(['Title']).agg(['mean', 'count']).sort_values(by=[('Survived', 'mean')], ascending=False))


def rank_algos(X_train, Y_train, X_test):
    classifiers = {'logreg': LogisticRegression(), 'svc': SVC(), 'knn': KNeighborsClassifier(),
                   'gaussian': GaussianNB(), 'perceptron': Perceptron(), 'linear_svc': LinearSVC(),
                   'sgd': SGDClassifier(), 'decision_tree': DecisionTreeClassifier(),
                   'random_forest': RandomForestClassifier(n_estimators=100),
    }
    predictions = {}
    scores = {}

    for classifier_key in classifiers:
        clf = classifiers[classifier_key]
        clf.fit(X_train, Y_train)
        predictions[classifier_key] = clf.predict(X_test)
        scores[classifier_key] = round(clf.score(X_train, Y_train) * 100, 2)

    models = pd.DataFrame({
        'Score': scores
    })

    submission = pd.DataFrame({
        'PassengerId': passengerIds,
        'Survived': predictions['random_forest']
    })

    submission.to_csv('./submission.csv', index=False)

    print(models.sort_values(by='Score', ascending=False))


'''
Processing
'''
# Do one-hot encoding for all categorical features (using dummy vars)
# Maybe combine SibSp and Parch into single feature?

# Create Title Feature
for key in df:
    df[key]['Title'] = df[key]['Name'].str.extract('([A-Za-z]+)\.', expand=False)

# Aggregate Titles
for data in df.values():
    data.replace(['Ms', 'Mme', 'Mlle'], 'Miss', inplace=True)
    data.replace(['Countess', 'Lady', 'Sir'], 'Royal', inplace=True)
    data.replace(['Col', 'Major'], 'Army', inplace=True)
    data.replace(['Capt', 'Don', 'Jonkheer', 'Rev'], 'Rare', inplace=True)

# Fill in missing Embarked
mode_embarked = df['train_df']['Embarked'].dropna().mode()[0]
for data in df.values():
    data['Embarked'].fillna(mode_embarked, inplace=True)

# Fill in missing Fare in test_df
df['test_df']['Fare'].fillna(df['test_df']['Fare'].dropna().median(), inplace=True)

for data in df.values():
    data.loc[data['SibSp'] >= 5, 'SibSp'] = 5
    data.loc[data['Parch'] >= 4, 'SibSp'] = 4
    data['FamilyMembers'] = data['SibSp'] + data['Parch'] + 1
    data.loc[data['FamilyMembers'] >= 8, 'SibSp'] = 8

# print(df['train_df'][['FamilyMembers', 'Survived']].groupby('FamilyMembers', as_index=False).mean().sort_values(by='Survived', ascending=False))

# Check which cols contain null (if any)
# print(df['train_df'].isnull().sum())
# print(df['test_df'].isnull().sum())
guess_ages = np.zeros([2, 3])
embarked_list = ['']
for data in df.values():
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = data[(data['Sex'] == i) & (data['Pclass'] == j + 1)]['Age'].dropna()

            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            data.loc[(data.Age.isnull()) & (data.Sex == i) & (data.Pclass == j + 1), 'Age'] = guess_ages[i, j]

        data['Age'] = df['Age'].astype('int')


for data in df.values():
    data.loc[data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[data['Age'] > 64, 'Age'] = 4

for key in df:
    df[key] = pd.get_dummies(df[key], columns=['Sex', 'Embarked', 'Pclass', 'Title', 'FamilyMembers', 'Age'])
    df[key].drop(['Cabin', 'Name', 'Ticket', 'PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True) # Also drop Cabin as we won't use it

df['train_df'], df['test_df'] = df['train_df'].align(df['test_df'], join='outer', axis=1, fill_value=0)
df['test_df'].drop('Survived', axis=1, inplace=True)

'''
Prediction
'''
X_train = df['train_df'].drop('Survived', axis=1)
Y_train = df['train_df']['Survived']
X_test = df['test_df'].copy()

rank_algos(X_train, Y_train, X_test)

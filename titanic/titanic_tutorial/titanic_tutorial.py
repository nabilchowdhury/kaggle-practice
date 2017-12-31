import pandas as pd
import numpy as np
import random as rnd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('../datasets/train.csv')
test_df = pd.read_csv('../datasets/test.csv')
combine = [train_df, test_df]

# Print available features
'''
Categorical: Survived, Sex, Embarked. Ordinal: Pclass
Numerical: Continuous: Age, Fare. Discrete: SibSp, Parch
'''
print(train_df.columns.values)
print(train_df.head())
print('-' * 100)
print(train_df.info())
print(test_df.info())
print('-' * 100)
print(train_df.describe(include=['O'])) # Feature distribution

'''
Drop features: Ticket, cabin, name, passengerId
'''
print('-' * 100)

pclass_survived = train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sex_survived = train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sibsp_survived = train_df[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
parch_survived = train_df[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

print(pclass_survived)
print(sex_survived)
print(sibsp_survived)
print(parch_survived)

'''
Visualize age
'''
# g = sns.FacetGrid(train_df, col='Survived')
# g.map(plt.hist, 'Age', bins=20)
# plt.show()

'''
Visualize pclass
'''
# grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

'''
Correlate categorical
'''
# grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
# grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
# grid.add_legend()
# plt.show()

'''
Correlate categorical and numeric
'''
# grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
# grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
# grid.add_legend()
# plt.show()

'''
Begin data processing
'''
print('-' * 100)
for df in combine:
    df.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

print(test_df.shape)

for df in combine:
    df['Title'] = df.Name.str.extract('([A-Za-z]+)\.', expand=False)

print(pd.crosstab(train_df['Title'], train_df['Sex']))

for df in combine:
    df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir',
                                       'Jonkheer', 'Dona'], 'Rare', inplace=True)
    df['Title'].replace('Mlle', 'Miss', inplace=True)
    df['Title'].replace('Ms', 'Miss', inplace=True)
    df['Title'].replace('Mme', 'Miss', inplace=True)

print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

title_mapping = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
for df in combine:
    df['Title'] = df['Title'].map(title_mapping)
    df['Title'] = df['Title'].fillna(0)

print(train_df.head())

test_df.drop('Name', axis=1, inplace=True)
train_df.drop(['Name', 'PassengerId'], axis=1, inplace=True)
print(train_df.shape)

sex_mapping = {'male': 0, 'female': 1}
for df in combine:
    df['Sex'] = df['Sex'].map(sex_mapping).astype('int')

print(train_df.head())

'''
Interpolate missing age
'''
# grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend()
# plt.show()

guess_ages = np.zeros([2, 3])

for df in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = df[(df['Sex'] == i) & (df['Pclass'] == j + 1)]['Age'].dropna()

            age_guess = guess_df.median()
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            df.loc[(df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j + 1), 'Age'] = guess_ages[i, j]

    df['Age'] = df['Age'].astype('int')

print(train_df.head())

'''
Create age bands
'''
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
print(train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean())

for df in combine:
    df.loc[df['Age'] <= 16, 'Age'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age'] = 3
    df.loc[df['Age'] > 64, 'Age'] = 4

train_df.drop(['AgeBand'], axis=1, inplace=True)
print(train_df.head())

'''
Define new FamilySize feature and drop SibSp and Parch
'''
for df in combine:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print(train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))

'''
Define IsAlone
'''
for df in combine:
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1

print(train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for df in combine:
    df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1, inplace=True)

print(train_df.head())

for df in combine:
    df['Age*Class'] = df['Age'] * df['Pclass']

print(train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10))
print(train_df[['Age*Class', 'Survived']].groupby(['Age*Class'], as_index=False).mean().sort_values(by='Survived', ascending=False))

freq_port = train_df.Embarked.dropna().mode()[0]
print(freq_port)

for df in combine:
    df['Embarked'].fillna(freq_port, inplace=True)

print(train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False))

embarked_map = {'S': 0, 'C': 1, 'Q': 2}
for df in combine:
    df['Embarked'] = df['Embarked'].map(embarked_map).astype('int')

print(train_df.head())

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
print(test_df.head())

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
print(train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='Survived', ascending=False))

for df in combine:
    df.loc[df['Fare'] <= 7.91, 'Fare'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare'] = 2
    df.loc[df['Fare'] > 31, 'Fare'] = 3
    df['Fare'] = df['Fare'].astype(int)

train_df.drop(['FareBand'], axis=1, inplace=True)

'''
Preprocessing finished
'''
print(train_df.head(10))
print(test_df.head(10))


'''
Prediction
'''
X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
print(X_train.shape, Y_train.shape, X_test.shape)

'''
Algo 1: Logistic Regression
'''
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
print(acc_log)

# Check feature correlation
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
print(coeff_df.sort_values(by='Correlation', ascending=False))

'''
Algo 2: Support Vector Machine
'''
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
print(acc_svc)

'''
Algo 3: KNN
'''
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
print(acc_knn)

'''
Algo 4: Gaussian Naive Bayes
'''
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)
print(acc_gaussian)

'''
Algo 5: Perceptron
'''
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
print(acc_perceptron)

'''
Algo 6: Linear SVC
'''
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

'''
Algo 7: Stochastic Gradient Descent
'''
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print(acc_sgd)

'''
Algo 8: Decision Tree
'''
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
print(acc_decision_tree)

'''
Algo 9: Random Forest
'''
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': Y_pred
})

submission.to_csv('./submission.csv', index=False)

'''
Rank algos
'''
models = pd.DataFrame({
    'Model': ['SVM', 'KNN', 'LogReg', 'Random Forest', 'Naive Bayes', 'Perceptron', 'SGD', 'Linear SVC', 'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian, acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]
})

print(models.sort_values(by='Score', ascending=False))
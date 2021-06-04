# Data Analysis and Wrangling Libraries
import pandas as pd
import numpy as np
import random as rnd

# Visualization Libraries
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# Machine Learning Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

print (train_df.columns.values)
print (train_df.head())
print (test_df.head())
print (train_df.tail())
print ()

print ("train_df.info()")
print (train_df.info())
print ('-'*40)

print ("test_df.info()")
print (test_df.info())
print ('-*'*40)

print (train_df.describe())
print ('-*'*40)

print (train_df.describe(include=['O']))
print ('-*'*40)

for x in ['Pclass','Sex','SibSp','Parch']:
    print (x)
    print (train_df[[x,'Survived']].groupby([x], as_index = False).mean().sort_values(by='Survived', ascending=False))
    print ('-*'*40)

# Histogram
print ("Histogram of 'Age' v/s 'Survived'")
g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)
#plt.show()
print ('-*'*40)

# Grid of histograms
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
#plt.show()
print ('-*'*40)

grid2 = sns.FacetGrid(train_df, row='Embarked', height=2.2, aspect=1.6)
grid2.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette = 'deep')
grid2.add_legend()
#plt.show()
print ('-*'*40)

grid3 = sns.FacetGrid(train_df, row='Embarked', col='Survived', height=2.2, aspect=1.6)
grid3.map(sns.barplot, 'Sex', 'Fare', alpha=0.5, ci=None)
grid3.add_legend()
#plt.show()
print ('-*'*40)

# Dropping two columns
print ("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
train_df = train_df.drop(['Ticket','Cabin'], axis=1)
test_df = test_df.drop(['Ticket','Cabin'], axis=1)
combine = [train_df, test_df]
print ("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)
print ('-*'*40)

# Add a column for 'Title'
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
print (pd.crosstab(train_df['Title'], train_df['Sex']))
print ('-*'*40)

# Replace French titles with English titles, to merge synonyms
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace('Mlle','Miss')
    dataset['Title'] = dataset['Title'].replace('Ms','Miss')
    dataset['Title'] = dataset['Title'].replace('Mme','Mrs')

# Merge rare titles into 'Rare'.
T = train_df['Title'].value_counts().to_dict()
print (T)

for dataset in combine:
    for t in T:
        if T[t]<10:
            dataset['Title'] = dataset['Title'].replace([t], 'Rare')

print (train_df[['Title','Survived']].groupby(['Title'], as_index=False).mean())
print ('-*'*40)

# Convert titles from categorical to ordinal.
title_mapping = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)
    dataset['Title'] = dataset['Title'].astype(int)

print ("Titles converted from categorical to ordinal.")
print (train_df.head())
print (test_df.head())
print ('-*'*40)

# Take out Name and PassengerId columns, because they don't have any predictive value.
print (train_df.columns.values)
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
print (train_df.columns.values)
print ('-*'*40)

# Convert Sex from categorical to ordinal
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female':1, 'male':0}).astype(int)
print (train_df.head(10))
print ('-*'*40)

# Fill in blank Age values
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=0.5, bins=20)
grid.add_legend()
#plt.show()
print ('-*'*40)

# Fill in missing ages with median value based on Age and Pclass.
guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range (0,2):
        for j in range (0,3):
            guess_df = dataset[(dataset['Sex'] == i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
            age_guess = guess_df.median()
            age_guess = int(age_guess/0.5+0.5)*0.5
            print (i,j+1, age_guess)
            dataset.loc[(dataset.Age.isnull()) & (dataset.Sex==i) & (dataset.Pclass==j+1), 'Age'] = age_guess
    dataset['Age'] = dataset['Age'].astype(int)
print(train_df.head(10))
print ('-*'*40)

# Create another column in train_df for AgeBand
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
T = train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
print (T)                           
print ('-*'*40)

# Replace Age with ordinals based on bands
for dataset in combine:
#    dataset['Age'] = dataset['Age'].map(lambda age:int((age-1)/16))
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

print ("Replace Age with ordinals based on bands")
print(train_df.head(10))
print ('-*'*40)



# Drop the AgeBand feature.
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
print(train_df.head(10))
print ('-*'*40)

# Create new feature combining existing features
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
T = train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print (T)
print ('-*'*40)

# Create a feature IsAlone.
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1
T = train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
print (T)
print ('-*'*40)

# Drop Parch, SibSp, and FamilySize features in favor of IsAlone
train_df = train_df.drop(['Parch','SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch','SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]
T = train_df.head(10)
print (T)
print ('-*'*40)

# Create a feature 'Age*Class'
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
T = train_df.loc[:, ['Age*Class','Age','Pclass']].head(10)
print (T)
print ('-*'*40)

# Complete the categorical feature Embarked by filling in with the most common occurrence.
freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
T = train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
print (T)
print ('-*'*40)

# Convert Embarked feature from categorical to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
T = train_df.head(10)
print (T)
print ('-*'*40)

# Complete the Fare feature in test_df
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
T = test_df.head(10)
print (T)
print ('-*'*40)

# Create 'FareBand' feature.
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
T = train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
print (T)
print ('-*'*40)

# Convert the 'Fare' feature to ordinal values based on the FareBand.
T = train_df.head(10)
print (T)
print ('-*'*40)
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[ (dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[ (dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
T = train_df.head(10)
print (T)
print ('-*'*40)
T = test_df.head(10)
print (T)
print ('-*'*40)

#################
#
# Finished with "3.  Wrangle, prepare, cleanse the data."
#   and "4.  Analyze, identify patterns, and explore the data."
#
# Now on to "5.  Model, predict, and solve the problem."
#
#############

X_train = train_df.drop('Survived', axis=1)
Y_train = train_df['Survived']
X_test = test_df.drop('PassengerId', axis=1).copy()
print (X_train.columns.values)
print (X_test.columns.values)
print (X_train.shape, Y_train.shape, X_test.shape)
print ('-*'*40)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train)*100,2)
print (acc_log)

coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df['Correlation'] = pd.Series(logreg.coef_[0])
T = coeff_df.sort_values(by='Correlation', ascending=False)
print (T)
print ('-*'*40)

T = T.to_latex()
write = open("../Crash_Data/03_10_Attempt/TitanicLogisticRegressionTable.tex", "w")
write.write("%s\n" % (T))
write.close()

########## Support Vector Machines
print ("Support Vector Machines")
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_train)
acc_svc = round(svc.score(X_train, Y_train)*100, 2)
print (acc_svc)
print ('-*'*40)

####### k-Nearest Neighbors
print ("k-Nearest Neighbors")
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train)*100,2)
print (acc_knn)
print ("-*"*40)

########### Gaussian Naive Bayes
print ("Gaussian Naive Bayes")
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train)*100, 2)
print (acc_gaussian)
print ("-*"*40)

######## Perceptron
print ("Perceptron")
algorithm = Perceptron()
algorithm.fit(X_train, Y_train)
Y_pred = algorithm.predict(X_test)
acc = round(algorithm.score(X_train, Y_train)*100,2)
print (acc)
acc_perceptron = acc
print ("-*"*40)

######## Linear Support Vector Classifier
print ("Linear SVC")
algorithm = LinearSVC()
algorithm.fit(X_train, Y_train)
Y_pred = algorithm.predict(X_test)
acc = round(algorithm.score(X_train, Y_train)*100,2)
print (acc)
acc_linear_svc = acc
print ("-*"*40)

######## Stochastic Gradient Descent Classifier
print ("SGDClassifier")
algorithm = SGDClassifier()
algorithm.fit(X_train, Y_train)
Y_pred = algorithm.predict(X_test)
acc = round(algorithm.score(X_train, Y_train)*100,2)
print (acc)
acc_sgd = acc
print ("-*"*40)

############ Decision Tree Classifier
print ("Decision Tree Classifier")
algorithm = DecisionTreeClassifier()
algorithm.fit(X_train, Y_train)
Y_pred = algorithm.predict(X_test)
acc = round(algorithm.score(X_train, Y_train)*100,2)
print (acc)
acc_decision_tree = acc
print ("-*"*40)

############ Random Forest Classifier
print ("Random Forest Classifier")
algorithm = RandomForestClassifier()
algorithm.fit(X_train, Y_train)
Y_pred = algorithm.predict(X_test)
acc = round(algorithm.score(X_train, Y_train)*100,2)
print (acc)
acc_random_forest = acc
print ("-*"*40)

###############
#
# Model Evaluation
#
##############

models = pd.DataFrame({
    'Model':[
        'Support Vector Machines',
        'KNN',
        'Logistic Regression',
        'Random Forest',
        'Naive Bayes',
        'Perceptron',
        'Stochastic Gradient Descent',
        'Linear SVC',
        'Decision Tree'
        ],
    'Score':[
        acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian,
        acc_perceptron, acc_sgd, acc_linear_svc, acc_decision_tree]})
T = models.sort_values(by='Score', ascending=False)
print (T)
print ('-*'*40)

T = T.to_latex()
write = open("../Crash_Data/03_10_Attempt/TitanicTable.tex", "w")
write.write("%s\n" % (T))

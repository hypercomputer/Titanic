import pandas as pd
import numpy as np
from sklearn import tree, preprocessing
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

def process_data(df):
    df = df.drop('Name', 1)
    df = df.drop('Ticket', 1)
    df = df.drop('PassengerId', 1)

    # transform the cabin entries into better labels
    df['Cabin'] = df['Cabin'].fillna('Z')
    df['Cabin'] = df['Cabin'].map(lambda x: x[0])

    #transform Sex to better labels
    df['Sex'] = df['Sex'].map(lambda x: x[0])

    # clean up and re-encode labels
    df = df.dropna()
    df['Cabin'] = preprocessing.LabelEncoder().fit_transform(df['Cabin'])
    df['Sex'] = preprocessing.LabelEncoder().fit_transform(df['Sex'])
    df['Embarked'] = preprocessing.LabelEncoder().fit_transform(df['Embarked'])
    return df


train = process_data(pd.read_csv('../datasets/train.csv', dtype={"Age": np.float64}, ))

X = np.array(train.drop('Survived', 1))
y = np.array(train['Survived'])

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)

test_set = pd.read_csv('../datasets/test.csv', dtype={"Age": np.float64}, ).merge(pd.read_csv('../datasets/gender_submission.csv'), on='PassengerId')
test_set = process_data(test_set)

X_test = np.array(test_set.drop('Survived', 1))
y_test = np.array(test_set['Survived'])

print('Decision Tree accuracy:\n', clf.score(X_test, y_test))
print('Cheating Decision Tree accuracy:\n', clf.score(X,y), '\n')

clf = GaussianNB()
clf = clf.fit(X,y)

print('Gaussian Naive Bayes accuracy:\n', clf.score(X_test, y_test))
print('Cheating Gaussian Naive Bayes accuracy:\n', clf.score(X,y), '\n')

clf = MultinomialNB()
clf = clf.fit(X,y)

print('Multinomial Naive Bayes accuracy:\n', clf.score(X_test, y_test))
print('Cheating Multinomial Naive Bayes accuracy:\n', clf.score(X,y), '\n')

clf = BernoulliNB()
clf = clf.fit(X,y)

print('Bernoulli Naive Bayes accuracy:\n', clf.score(X_test, y_test))
print('Cheating Bernoulli Naive Bayes accuracy:\n', clf.score(X,y), '\n')

clf = RandomForestClassifier()
clf = clf.fit(X,y)

print('Random Forest accuracy:\n', clf.score(X_test, y_test))
print('Cheating Random Forest accuracy:\n', clf.score(X,y), '\n')

clf = svm.SVC()
clf = clf.fit(X,y)

print('C-Support Vector Classification accuracy:\n', clf.score(X_test, y_test))
print('Cheating C-Support Vector Classification accuracy:\n', clf.score(X,y), '\n')

clf = svm.LinearSVC()
clf = clf.fit(X,y)

print('Linear Support Vector Classification accuracy:\n', clf.score(X_test, y_test))
print('Cheating Linear Support Vector Classification accuracy:\n', clf.score(X,y), '\n')

clf = linear_model.LogisticRegression(penalty='l2')
clf = clf.fit(X,y)

print('Linear Regression accuracy:\n', clf.score(X_test, y_test))
print('Cheating Linear Regression accuracy:\n', clf.score(X,y), '\n')

clf = MLPClassifier()
clf = clf.fit(X,y)

print('Neural Network accuracy:\n', clf.score(X_test, y_test))
print('Neural Network  accuracy:\n', clf.score(X,y), '\n')

print(test_set.head())
print(test_set[(test_set['Survived'] == 0) & (test_set['Sex'] == 1)])

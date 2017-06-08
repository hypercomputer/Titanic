import pandas as pd
import numpy as np

def gender_survival(df, gender, survived):
    return len(df[(df['Sex'] == gender) & (df['Survived'] == survived)])/len(df[df['Sex'] == gender])

train = pd.read_csv('../datasets/train.csv', dtype={"Age": np.float64}, )
train = train[['Survived', 'Sex', 'Age']]
train = train.dropna()

print('Female stats:')
print('died:', gender_survival(train, 'female', 0))
print('lived:', gender_survival(train, 'female', 1), '\n')

print('Male stats:')
print('died:', gender_survival(train, 'male', 0))
print('lived:', gender_survival(train, 'male', 1), '\n')
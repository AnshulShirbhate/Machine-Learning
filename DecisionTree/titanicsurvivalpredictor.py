import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Code to drop the unwanted features
# df = pd.read_csv('Titanic-Dataset.csv')
# df.drop(['Name', 'Ticket', 'Embarked', 'PassengerId'], axis='columns', inplace=True)
# df.to_csv('titanic_cleaned_data.csv', index=False)

df = pd.read_csv('titanic_cleaned_data.csv')


# Code to clean and transform the data
categorical_cols = ['Sex', 'Cabin']
for col in categorical_cols:
    df[col] = df[col].fillna('Missing')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


df['Age'] = df['Age'].fillna(df['Age'].median())

X = df.drop(columns='Survived')
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc_score = accuracy_score(y_test, y_pred)
print(acc_score)


# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier as dtc
# from sklearn.metrics import accuracy_score

# df = pd.read_csv('Cleaned-Dataset.csv')


# # Selecting the Input set and the Output set
# X = df.drop(columns='Survived')
# y = df['Survived']

# # Split the data into training and testing set
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = dtc()

# model.fit(x_test, x_train)

# predictions = model.predict(x_test)

# score = accuracy_score(y_test, predictions)
# print(score)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('Cleaned-Dataset.csv')
# print(df.columns)
# Handle categorical data
categorical_cols = ['Sex', 'Cabin']

for col in categorical_cols:
    df[col] = df[col].fillna('Missing')  # Handle missing categorical values
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Encode categories as numbers

# Handle missing numerical data
df['Age'] = df['Age'].fillna(df['Age'].median())  # Replace missing Age with median

# Selecting the Input set and the Output set
X = df.drop(columns='Survived')
y = df['Survived']

# Split the data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = dtc(random_state=42)
model.fit(x_train, y_train)

# Make predictions
# predictions = model.predict(x_test)

# # Evaluate the model
# score = accuracy_score(y_test, predictions)
# print(f"Accuracy Score: {score:.2f}")

predictions = model.predict([[2, 1, 21, 1, 2, 71, 1]])
print(predictions)
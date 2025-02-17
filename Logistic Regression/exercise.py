import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
import pickle

df = load_iris()
X = df.data
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model = LogisticRegression()
# model.fit(X_train, y_train)

# with open('model', 'wb') as f:
#     pickle.dump(model, f)
with open('model', 'rb') as f:
    model = pickle.load(f)

# 5.1 3.5 1.4 0.2 // 0
# 6.3 3.3 6.  2.5 // 2

# print(model.predict([[6.3, 3.3, 6, 2.5]]))

# Accuracy
# accuracy = model.score(X_train, y_train)

y_predicted = model.predict(X_test)
accuracy = accuracy_score(y_test, y_predicted)
print(accuracy)
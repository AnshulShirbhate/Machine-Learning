import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

digits = load_digits()
# print(digits.data[0])
# plt.imshow(digits.images[0], cmap='gray')
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
model = LogisticRegression()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
y_predicted = model.predict(X_test)

accuracy = accuracy_score(y_test, y_predicted)
print(accuracy)
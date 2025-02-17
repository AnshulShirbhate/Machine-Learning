import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

digits_dataset = load_digits()
df = pd.DataFrame(digits_dataset.data, columns=digits_dataset.feature_names)

model = SVC()

X = df
y = digits_dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
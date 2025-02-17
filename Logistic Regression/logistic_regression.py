import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('insurance_data.csv')
plt.scatter(df.age, df.bought_insurance, marker='+',)

X = df[['age']]
y = df['bought_insurance']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression()

model.fit(X_train, y_train)

ans = model.predict(X_test)
accuracy = model.score(X_train, y_train)
print(accuracy)
# plt.show()
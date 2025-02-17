import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split


df = pd.read_csv('carprices.csv')

print(df)
X = df[['Mileage', 'Age(yrs)']]
y = df['Sell Price($)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = linear_model.LinearRegression()

model.fit(X_train, y_train)

# ans = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(accuracy)
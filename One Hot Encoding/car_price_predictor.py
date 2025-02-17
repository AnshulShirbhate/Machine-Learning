import pandas as pd
from sklearn import linear_model
import pickle

df = pd.read_csv('carprices.csv')
# dummies = pd.get_dummies(df['Car Model'])

# merged = pd.concat([df, dummies], axis='columns')
# final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis='columns')


# model = linear_model.LinearRegression()

# X = final.drop(['Sell Price($)'], axis='columns')
# y = final['Sell Price($)'] 
# print(X)

# model.fit(X, y)

# # with open('model', 'wb') as f:
# #     pickle.dump(model, f)
# # ans = model.predict([[69000, 6, 0 ,1]])
# accuracy = model.score(X, y)
# print(accuracy)




# Now doing the same thing with sklearn Onehot encoding libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

dfle = df
dfle['Car Model'] = le.fit_transform(dfle['Car Model'])

X = dfle[['Car Model', 'Mileage', 'Age(yrs)']].values
y = dfle['Sell Price($)']

ohe = OneHotEncoder(categorical_features=[0])
ohe.fit_transform(X).toarray()

print(X)
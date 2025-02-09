# We have to predict the prices of homes based on multiple factors such as area, bedroom and the age of the house.


import pandas as pd
from sklearn import linear_model
import math

df = pd.read_csv('home_prices.csv')

# As there are missing values in the bedroom feature/column we are going to fill the void spaces with the median of the column values
median_bedrooms = math.floor(df.bedroom.median())
df.bedroom.fillna(median_bedrooms ,inplace=True)

model = linear_model.LinearRegression()
model.fit(df[['area', 'bedroom', 'age']], df.price)
# print(model.coef_)

output = model.predict([[3000, 3, 40]])
print(output)

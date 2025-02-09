import pandas as pd
from sklearn import linear_model
from word2number import w2n
import math

df = pd.read_csv('employee_data.csv')

df.experience.fillna('zero', inplace=True)
df['experience']= df['experience'].apply(w2n.word_to_num)
median_test_score = math.floor(df['test_score(out of 10)'].median())
df['test_score(out of 10)'].fillna(median_test_score, inplace=True)


model = linear_model.LinearRegression()
model.fit(df[['experience', 'test_score(out of 10)', 'interview_score(out of 10)']], df['salary($)'])

output = model.predict([[2, 10, 10]])
print(output)
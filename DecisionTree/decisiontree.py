import pandas as pd
from sklearn.preprocessing import LabelEncoder # This library is used to convert the character or text data into numeric format
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('salaries.csv')

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


# In these 3 lines of code we are adding the numeric transformed features of the company name, job name and degree name.
df['company_n'] = le_company.fit_transform(df.company)
df['job_n'] = le_job.fit_transform(df.job)
df['degree_n'] = le_degree.fit_transform(df.degree)

df = df.drop(['company', 'job', 'degree'], axis='columns') # Here we are dropping the columns containing text data

X = df.drop('salary_more_then_100k', axis='columns')
y = df['salary_more_then_100k']

model = DecisionTreeClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)

print(model.score(X, y))



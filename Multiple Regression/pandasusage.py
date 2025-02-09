
import pandas as pd

file = pd.read_csv("/content/Spotify100.csv")

df = pd.DataFrame(file)

print(df)

print(df.head(5))

print(df.tail(5))

print(df.isnull())

print(df.dropna())

print(df.describe())

df.notnull()

df.replace()




 OR




import pandas as pd import numpy as np
import matplotlib.pyplot as plt
# Load dataset (replace 'data.csv' with your dataset) df = pd.read_csv('/content/data.csv')
# Handling missing values
# Using isnull() and notnull() to detect missing values print("Missing Values:\n", df.isnull().sum())
print("Non-Missing Values:\n", df.notnull().sum()) # Using dropna() to drop rows with missing values cleaned_df = df.dropna()
# Using fillna() to fill missing values with a specified value filled_df = df.fillna(0) # Fill missing values with 0
# Using replace() to replace blank textual data with 'zzz' # Replace blanks in 'category' column with 'zzz'
df['category'] = df['category'].fillna('zzz')
# Using interpolate() to interpolate missing numerical data interpolated_df = df.interpolate()
# Data Visualization # Bar graph
df['category'].value_counts().plot(kind='bar') plt.title('Bar Graph')
plt.xlabel('Category') plt.ylabel('Count')
plt.show()
# Scatterplot
plt.scatter(df['x'], df['y']) plt.title('Scatterplot')
plt.xlabel('X')
plt.ylabel('Y') plt.show()
# Line plot
df_cleaned = df.dropna(subset=['value']) # Drop rows with NaN values in 'value' column plt.plot(df_cleaned['time'], df_cleaned['value'])




plt.title('Line Plot') plt.xlabel('Time') plt.ylabel('Value')
 
plt.show()

plt.title('Line Plot') plt.xlabel('Time') plt.ylabel('Value')

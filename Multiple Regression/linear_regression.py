import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])


# print(diabetes.DESCR) // To get information about the dataset


diabetes_x = diabetes.data[:, np.newaxis, 3]

# diabetes_x = diabetes.data #To use all features in the code

print(diabetes.feature_names)

# diabetes_x_train = diabetes_x[:-30]
# diabetes_x_test = diabetes_x[-30:]

# diabetes_y_train = diabetes.target[:-30]
# diabetes_y_test = diabetes.target[-30:]

x_train, x_test, y_train, y_test = train_test_split(diabetes_x, diabetes.target, test_size=0.2)


model = linear_model.LinearRegression()
model.fit(x_train, y_train)

diabetes_y_predicted = model.predict(x_test)

print("Mean squared error is: ", mean_squared_error(y_test, diabetes_y_predicted))

plt.scatter(x_test, y_test)
plt.plot(x_test, diabetes_y_predicted)

plt.show()
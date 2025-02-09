# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# # Step 1: Generate a Sample Dataset
# np.random.seed(42)
# n_samples = 200

# # Create a sample dataset with parameters Age, BMI, Weight, BP, Height, and a binary target variable
# data = pd.DataFrame({
#     'Age': np.random.randint(20, 70, size=n_samples),
#     'BMI': np.random.uniform(18, 35, size=n_samples),
#     'Weight': np.random.uniform(50, 100, size=n_samples),
#     'BP': np.random.uniform(80, 180, size=n_samples),
#     'Height': np.random.uniform(150, 200, size=n_samples),
#     'Target': np.random.randint(0, 2, size=n_samples)  # Binary classification target
# })

# # Step 2: Split the Data into Training and Testing Sets
# X = data.drop('Target', axis=1)
# y = data['Target']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Step 3: Train the SVM Classifier
# svm = SVC(kernel='linear', random_state=42)
# svm.fit(X_train, y_train)

# # Step 4: Make Predictions and Evaluate the Model
# y_pred = svm.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Accuracy: {accuracy:.2f}")
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

# # Confusion Matrix
# conf_matrix = confusion_matrix(y_test, y_pred)
# plt.figure(figsize=(6, 4))
# sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
# plt.title('Confusion Matrix')
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.show()

# # Step 5: Visualize the Results (Only possible for 2D data)
# # For visualization purposes, let's reduce the dataset to 2D
# data_2d = data[['Age', 'BMI', 'Target']]
# X_2d = data_2d.drop('Target', axis=1)
# y_2d = data_2d['Target']
# X_train_2d, X_test_2d, y_train_2d, y_test_2d = train_test_split(X_2d, y_2d, test_size=0.3, random_state=42)

# # Train the SVM on 2D data
# svm_2d = SVC(kernel='linear', random_state=42)
# svm_2d.fit(X_train_2d, y_train_2d)

# # Plot decision boundary
# def plot_decision_boundary(clf, X, y):
#     plt.figure(figsize=(10, 6))
#     x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
#     y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
#     Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)
#     plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
#     plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolor='k')
#     plt.xlabel('Age')
#     plt.ylabel('BMI')
#     plt.title('SVM Decision Boundary')
#     plt.show()

# plot_decision_boundary(svm_2d, X_test_2d, y_test_2d)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Generate a Sample Dataset
np.random.seed(42)
n_samples = 200

# Create a sample dataset with parameters Age, BMI, Weight, BP, Height, and a binary target variable
data = pd.DataFrame({
    'Age': np.random.randint(20, 70, size=n_samples),
    'BMI': np.random.uniform(18, 35, size=n_samples),
    'Weight': np.random.uniform(50, 100, size=n_samples),
    'BP': np.random.uniform(80, 180, size=n_samples),
    'Height': np.random.uniform(150, 200, size=n_samples),
    'Target': np.random.randint(0, 2, size=n_samples)  # Binary classification target
})

# Use a 2D subset of the data for visualization (Age and BMI)
data_2d = data[['Age', 'BMI', 'Target']]
X = data_2d.drop('Target', axis=1)
y = data_2d['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Train the SVM Classifier
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# Step 3: Make Predictions and Evaluate the Model
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Step 4: Visualize the Results with Hyperplane
def plot_decision_boundary(clf, X, y):
    plt.figure(figsize=(10, 6))
    x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
    y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, s=20, edgecolor='k')
    
    # Plot the hyperplane
    coef = clf.coef_[0]
    intercept = clf.intercept_[0]
    slope = -coef[0] / coef[1]
    intercept = -intercept / coef[1]
    x_vals = np.linspace(x_min, x_max)
    y_vals = slope * x_vals + intercept
    plt.plot(x_vals, y_vals, 'k--')

    plt.xlabel('Age')
    plt.ylabel('BMI')
    plt.title('SVM Decision Boundary with Hyperplane')
    plt.show()

plot_decision_boundary(svm, X_test, y_test)

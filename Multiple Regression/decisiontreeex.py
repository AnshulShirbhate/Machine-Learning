import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Generate a Sample Dataset
np.random.seed(42)
n_samples = 200
n_features = 5

# Create a sample dataset with parameters Age, BMI, Weight, BP, Height, and a binary target variable
data = pd.DataFrame({
    'Age': np.random.randint(20, 70, size=n_samples),
    'BMI': np.random.uniform(18, 35, size=n_samples),
    'Weight': np.random.uniform(50, 100, size=n_samples),
    'BP': np.random.uniform(80, 180, size=n_samples),
    'Height': np.random.uniform(150, 200, size=n_samples),
    'Target': np.random.randint(0, 2, size=n_samples)  # Binary classification target
})

# Step 2: Split the Data into Training and Testing Sets
X = data.drop('Target', axis=1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 4: Make Predictions and Evaluate the Model
y_pred = clf.predict(X_test)
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

# Step 5: Visualize the Decision Tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=['Class 0', 'Class 1'], filled=True)
plt.title('Decision Tree Visualization')
plt.show()

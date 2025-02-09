import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Sample data creation
np.random.seed(42)  # For reproducibility
data_size = 1000
data = pd.DataFrame({
    'parameter': np.random.randn(data_size),  # Normally distributed data
    'label': np.random.randint(0, 2, data_size)  # Binary labels for example
})

# Split the data
train_data, temp_data = train_test_split(data, test_size=0.4, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(validation_data)}")
print(f"Testing set size: {len(test_data)}")

# Output the first few rows of each set
print("Training set sample:")
print(train_data.head())

print("Validation set sample:")
print(validation_data.head())

print("Testing set sample:")
print(test_data.head())

# Plotting the normal distribution
mu, sigma = 0, 1  # Mean and standard deviation
x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
y = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

plt.plot(x, y, label='Normal Distribution (mean=0, std=1)')
plt.title('Normal Distribution')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()

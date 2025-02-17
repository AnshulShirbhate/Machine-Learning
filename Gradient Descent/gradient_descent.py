# In this we need to reduce the cost at each iteration to find the global minima. To do so we can experiment with adjusting the values
# of the iterations, the learning rate etc.

import numpy as np

def gradient_descent(x, y):
    # Start from having the coefficients value as 0 and intercept's value as 0
    m_curr = b_curr = 0

    # Decide the number of iterations
    # iterations = 1000
    # iterations = 10
    iterations = 1000
    n = len(x)
    # learning_rate = 0.001
    # learning_rate = 0.1
    learning_rate = 0.08

    for i in range(iterations):
        # Formula to find the y_predicted i.e. y=mx+b
        y_predicted = m_curr * x + b_curr

        # Cost function formula or Formula to calculate the cost or MSE
        cost = (1/n) * sum([val**2 for val in ( y - y_predicted ) ])

        # Derivative of m (coefficient)
        md = -(2/n)*sum(x*(y-y_predicted))
        # Derivative of m (intercept)
        bd = -(2/n)*sum(y-y_predicted)

        # Formula to update the values of m and b for further calcuation to reach the global minima 
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print(f"m {m_curr}, b {b_curr}, cost {cost}, i {i}")

x = np.array([1, 2, 3, 4 ,5])
y = np.array([5, 7, 9, 11, 13])
gradient_descent(x, y)

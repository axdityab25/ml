import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset (change the file name and columns as needed for each question)
# Example: Replace 'Advertising Budget and Sales.csv' with the appropriate CSV file for each question.
data = pd.read_csv('Advertising Budget and Sales.csv')  # Replace with the correct dataset file

# Select feature (independent variable) and target (dependent variable)
# For example, question 1 might require 'Radio Ad Budget ($)' as the feature.
x = data['Radio Ad Budget ($)'].values  # Replace with the appropriate feature column
y = data['Sales ($)'].values  # Replace with the appropriate target column

# Normalize the feature for gradient descent to improve performance
x = (x - np.mean(x)) / np.std(x)

# Initialize parameters
m = 0  # Initial slope
c = 0  # Initial intercept
L = 0.001  # Learning rate
epochs = 100000  # Number of iterations
n = float(len(x))  # Number of data points
cost_history = []  # List to store the cost at each epoch

# Gradient Descent Loop
for i in range(epochs):
    # Prediction
    y_pred = m * x + c
    
    # Cost function (Mean Squared Error)
    cost = (1/n) * np.sum((y - y_pred) ** 2)
    cost_history.append(cost)
    
    # Calculate gradients
    d_m = (-2/n) * np.sum(x * (y - y_pred))  # Gradient of m
    d_c = (-2/n) * np.sum(y - y_pred)  # Gradient of c
    
    # Update parameters
    m = m - L * d_m  # Update slope
    c = c - L * d_c  # Update intercept

# Output the results
print(f"The slope (m) is: {m}")
print(f"The intercept (c) is: {c}")

# Calculate Root Mean Square Error (RMSE)
rmse = np.sqrt(np.mean(((m * x + c) - y) ** 2))
print(f"RMSE: {rmse}")

# Plotting
plt.figure(figsize=(10, 5))

# Plotting the data and the regression line
plt.subplot(1, 2, 1)
plt.scatter(x, y, color='blue', label="Actual Data")
plt.plot(x, m * x + c, color='red', label="Predicted Line")
plt.xlabel("Feature (e.g., Radio Ad Budget)")
plt.ylabel("Target (e.g., Sales)")
plt.title("Feature vs Target")
plt.legend()

# Plotting the cost function over epochs
plt.subplot(1, 2, 2)
plt.plot(range(epochs), cost_history, color="green")
plt.xlabel("Number of Iterations")
plt.ylabel("Cost")
plt.title('Cost Function Over Time')
plt.show()

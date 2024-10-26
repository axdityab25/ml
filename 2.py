import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('Advertising Budget and Sales.csv')

# Drop missing values and duplicates
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)

# Extract the feature columns and target column
X1 = data['Radio Ad Budget ($)'].values
X2 = data['Newspaper Ad Budget ($)'].values
Y = data['Sales ($)'].values

# Number of data points
n = len(X1)

# Initialize parameters for the linear regression (slope and intercept)
a = 0.0  # Coefficient for Radio Ad Budget
b = 0.0  # Coefficient for Newspaper Ad Budget
c = 0.0  # Intercept

# Hyperparameters
epochs = 5000  # Number of iterations
L = 0.0001  # Learning rate

# List to track the cost function value over time
cost_history = []

# Perform Gradient Descent
for i in range(epochs):
    # Predicted value of Y
    Y_pred = a * X1 + b * X2 + c
    
    # Calculate the cost (Mean Squared Error)
    cost_function = (1/n) * np.sum((Y - Y_pred)**2)
    cost_history.append(cost_function)

    # Calculate gradients
    D_a = (-2/n) * np.sum((Y - Y_pred) * X1)  # Partial derivative w.r.t. a
    D_b = (-2/n) * np.sum((Y - Y_pred) * X2)  # Partial derivative w.r.t. b
    D_c = (-2/n) * np.sum(Y - Y_pred)         # Partial derivative w.r.t. c
    
    # Update parameters using the gradients
    a = a - L * D_a
    b = b - L * D_b
    c = c - L * D_c

# Make final predictions after training
Y_pred = a * X1 + b * X2 + c

# Calculate RMSE (Root Mean Squared Error)
rmse = np.sqrt(np.mean((Y - Y_pred)**2))
print(f"RMSE: {rmse}")

# Print the actual and predicted values
print(f"Actual Y: {Y}")
print(f"Predicted Y: {Y_pred}")

# Plot the cost function over iterations
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), cost_history, color="green")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.title("Cost Function Over Time")
plt.show()
import numpy as np
import pandas as pd

# 1. Sigmoid Function (hypothesis)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 2. Cost Function and Gradient Descent
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    cost = -(1/m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return cost

def gradient_descent(X, y, theta, alpha, num_iterations):
    m = len(y)
    cost_history = []
    
    for i in range(num_iterations):
        h = sigmoid(np.dot(X, theta))
        gradient = (1/m) * np.dot(X.T, (h - y))  # Compute the gradient
        theta = theta - alpha * gradient  # Update theta
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

        # Optional: Print the cost every 100 iterations to monitor the learning process
        if i % 100 == 0:
            print(f"Cost after iteration {i}: {cost}")
    
    return theta, cost_history

# 3. Prediction Function
def predict(X, theta):
    return sigmoid(np.dot(X, theta)) >= 0.5  # Threshold at 0.5

# 4. Logistic Regression (Putting everything together)
def logistic_regression(X, y, alpha=0.01, num_iterations=1000):
    X = np.c_[np.ones((X.shape[0], 1)), X]  # Add intercept term (bias)
    theta = np.zeros(X.shape[1])  # Initialize theta to zeros
    theta, cost_history = gradient_descent(X, y, theta, alpha, num_iterations)
    return theta

# 5. Load the Iris dataset
df = pd.read_csv(r'C:\Users\ASUS\Desktop\vighnesh\ML\Iris\Iris.csv')

# Selecting two features and converting the target to a binary problem
# For this example, let's use 'SepalLengthCm' and 'SepalWidthCm' as features.
# Convert the target to 1 for 'Iris-setosa' and 0 for all other species.
df['target'] = np.where(df['Species'] == 'Iris-setosa', 1, 0)

# Use 'SepalLengthCm' and 'SepalWidthCm' as features
X = df[['SepalLengthCm', 'SepalWidthCm']].values  # Features
y = df['target'].values  # Target (0 or 1)

# 6. Split the dataset into training and test sets (80% train, 20% test)
train_size = int(0.7 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 7. Train the Logistic Regression model
theta = logistic_regression(X_train, y_train, alpha=0.01, num_iterations=1000)

# 8. Make predictions on the test set
X_test_with_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]  # Add intercept term
y_pred = predict(X_test_with_bias, theta)

# 9. Calculate accuracy
accuracy = np.mean(y_pred == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Optional: Evaluate the confusion matrix
confusion_matrix = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
print("\nConfusion Matrix:")
print(confusion_matrix)
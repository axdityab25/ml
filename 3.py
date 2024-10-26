# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load and explore the dataset
# Assuming the dataset is in a CSV file named 'Social_Network_Ads.csv'
data = pd.read_csv('Social_Network_Ads.csv')

# Step 3: Data Preprocessing
# Convert Gender to numerical (0 for Female, 1 for Male)
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'].values)

# Features (X) and target (y)
X = data[['Age', 'EstimatedSalary']].values
y = data['Purchased'].values

# Step 4: Split the dataset into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Scaling (important for SVM)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train an SVM classifier
# Change kernel='linear' to 'rbf' for questions 22-26
svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = svm_classifier.predict(X_test)

# Step 8: Evaluate the model
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# For questions requiring other metrics
# Precision
precision = precision_score(y_test, y_pred)
print(f"Precision: {precision * 100:.2f}%")

# F1-score
f1 = f1_score(y_test, y_pred)
print(f"F1 Score: {f1 * 100:.2f}%")

# Step 9: Visualize the Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

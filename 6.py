# Step 1: Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# Step 2: Load and Understand the Dataset
iris = load_iris()
X = iris.data
y = iris.target

print("Features:", iris.feature_names)
print("Classes:", iris.target_names)

# Step 3: Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build a Classifier without PCA
rf_classifier = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier.fit(X_train, y_train)

# Step 5: Predict and Evaluate without PCA
y_pred = rf_classifier.predict(X_test)
accuracy_no_pca = accuracy_score(y_test, y_pred)
print(f"Accuracy without PCA: {accuracy_no_pca * 100:.2f}%")

# Step 6: Apply PCA for Dimensionality Reduction
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Step 7: Build a Classifier with PCA-transformed Data
rf_classifier_pca = RandomForestClassifier(n_estimators=50, random_state=42)
rf_classifier_pca.fit(X_train_pca, y_train)

# Step 8: Predict and Evaluate with PCA
y_pred_pca = rf_classifier_pca.predict(X_test_pca)
accuracy_with_pca = accuracy_score(y_test, y_pred_pca)
print(f"Accuracy with PCA: {accuracy_with_pca * 100:.2f}%")

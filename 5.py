# Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris Dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
print("Decision Tree Classifier")
clf = DecisionTreeClassifier(criterion='gini', max_depth=4, min_samples_split=2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Decision Tree classifier: {accuracy:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Diagram")
plt.show()


# Bagging Classifier
print("Bagging Classifier")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)
bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=32)
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Bagging classifier: {accuracy * 100:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Random Forest Classifier
print("Random Forest Classifier")
rfc = RandomForestClassifier(n_estimators=10, random_state=32)
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Random Forest classifier: {accuracy * 100:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Gradient Boost Classifier
print("Gradient Boost Classifier")
gbc = GradientBoostingClassifier(n_estimators=10, random_state=32)
gbc.fit(X_train, y_train)
y_pred = gbc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the Gradient Boost classifier: {accuracy * 100:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# AdaBoost Classifier
print("AdaBoost Classifier")
abc = AdaBoostClassifier(n_estimators=10, random_state=37)
abc.fit(X_train, y_train)
y_pred = abc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy of the AdaBoost Classifier: {accuracy * 100:.2f}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

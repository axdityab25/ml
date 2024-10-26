# Step 1: Load Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=37)

# Function to train, predict, and evaluate a classifier
def evaluate_classifier(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy of {model_name}: {accuracy * 100:.2f}%')
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 1. Decision Tree Classifier
print("\nDecision Tree Classifier (Criterion: 'log_loss')")
decision_tree = DecisionTreeClassifier(criterion='log_loss', max_depth=4, random_state=42)
evaluate_classifier(decision_tree, "Decision Tree")

# Visualize the Decision Tree
plt.figure(figsize=(12, 8))
plot_tree(decision_tree, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Decision Tree Diagram")
plt.show()

# 2. Bagging Classifier
print("\nBagging Classifier")
bagging_model = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=10, random_state=32)
evaluate_classifier(bagging_model, "Bagging Classifier")

# 3. Random Forest Classifier
print("\nRandom Forest Classifier")
random_forest = RandomForestClassifier(n_estimators=10, random_state=32)
evaluate_classifier(random_forest, "Random Forest Classifier")

# 4. Gradient Boosting Classifier
print("\nGradient Boosting Classifier")
gradient_boosting = GradientBoostingClassifier(n_estimators=10, random_state=32)
evaluate_classifier(gradient_boosting, "Gradient Boosting Classifier")

# 5. AdaBoost Classifier
print("\nAdaBoost Classifier")
ada_boost = AdaBoostClassifier(n_estimators=10, random_state=32)
evaluate_classifier(ada_boost, "AdaBoost Classifier")

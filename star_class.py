import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("star_classification.csv")
print(data.shape)
print(data["class"].value_counts())

# Features and labels
features = ["u", "g", "r", "i", "z", "redshift"]
X = data[features]
y = data["class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
print("🌌 COSMIC CLASSIFIER")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(y_test, predictions))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_test, predictions)
plt.title("Star, Galaxy and Quasar Classification")
plt.show()

# Feature importance
importances = pd.Series(model.feature_importances_, index=features).sort_values()
importances.plot(kind="barh", title="Feature Importance")
plt.tight_layout()
plt.show()
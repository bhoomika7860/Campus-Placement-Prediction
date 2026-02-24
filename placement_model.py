
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score


# Load dataset
df = pd.read_csv("Placement_Data_Full_Class.csv")
# Increase dataset size using sampling

# Drop unnecessary column
df.drop("sl_no", axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
categorical_cols = ["gender", "ssc_b", "hsc_b", "hsc_s",
                    "degree_t", "workex", "specialisation", "status"]

for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Split data into features and target
X = df.drop(["status", "salary"], axis=1)
y = df["status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from sklearn.utils import resample

# Combine X_train and y_train
train_data = pd.concat([X_train, y_train], axis=1)

# Separate classes
placed = train_data[train_data.status == 1]
not_placed = train_data[train_data.status == 0]

# Oversample minority class
not_placed_upsampled = resample(
    not_placed,
    replace=True,
    n_samples=700,
    random_state=42
)

# Combine back
train_upsampled = pd.concat([placed, not_placed_upsampled])

# Split again
X_train = train_upsampled.drop("status", axis=1)
y_train = train_upsampled["status"]

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Accuracy
print("Model Accuracy:", accuracy_score(y_test, predictions))
from sklearn.metrics import classification_report, confusion_matrix

print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))
rf_probs = model.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_probs)
print("Random Forest AUC:", rf_auc)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
print("\nLogistic Regression Report:\n", classification_report(y_test, lr_pred))
lr_probs = lr.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_probs)
print("Logistic Regression AUC:", lr_auc)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("\nDecision Tree Report:\n", classification_report(y_test, dt_pred))
dt_probs = dt.predict_proba(X_test)[:, 1]
dt_auc = roc_auc_score(y_test, dt_probs)
print("Decision Tree AUC:", dt_auc)

# SVM
svm = SVC(probability=True)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
print("\nSVM Report:\n", classification_report(y_test, svm_pred))
svm_probs = svm.predict_proba(X_test)[:, 1]
svm_auc = roc_auc_score(y_test, svm_probs)
print("SVM AUC:", svm_auc)

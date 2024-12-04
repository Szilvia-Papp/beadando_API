import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.tracking import MlflowClient

#%% Load data
data = pd.read_csv("./data/heart_statlog_cleveland_hungary_final.csv", sep=",")

# Preprocess data
data['sex'].replace({1: 'Male', 0: 'Female'}, inplace=True)
data['chest pain type'].replace({1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'}, inplace=True)
data['fasting blood sugar'].replace({1: 'bs > 120 mg/dl', 0: 'bs <= 120 mg/dl'}, inplace=True)
data['resting ecg'].replace({1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy', 0: 'normal'}, inplace=True)
data['exercise angina'].replace({1: 'exercise angina', 0: 'no exercise angina'}, inplace=True)
data['ST slope'].replace({1: 'upsloping', 2: 'flat', 3: 'downsloping'}, inplace=True)
data['target'].replace({1: 'heart disease', 0: 'Normal'}, inplace=True)

# Drop redundant columns and encode categorical variables
data = pd.get_dummies(data)
data.drop(columns=["sex_Female", "chest pain type_asymptomatic", "target_Normal",
                   "fasting blood sugar_bs <= 120 mg/dl", "ST slope_upsloping",
                   "exercise angina_no exercise angina", "resting ecg_normal"], inplace=True)

# Define features and target
X = data.drop(['target_heart disease'], axis=1)
y = data['target_heart disease']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#%% Train logistic regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)  # Train the model

# Evaluate the model
y_pred = logreg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

#%% Mlflow
# Save the model
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Specify server
client = MlflowClient("http://127.0.0.1:5000") 
name = "heart_disease_logreg"  # Experiment name

try:
    client.create_experiment(name)
except Exception as e:
    pass  # Experiment might already exist

experiment_id = client.get_experiment_by_name(name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    run_id = mlflow.active_run().info.run_id
    mlflow.sklearn.log_model(logreg, "model")  # Log the trained model
    mlflow.log_param("input", X_train.columns.to_list())  # Log input features
    mlflow.log_metric("accuracy", accuracy)  # Log metrics
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)

print(f"Model logged with run ID: {run_id}")
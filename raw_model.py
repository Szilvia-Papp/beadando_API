import numpy as np
import pandas as pd

#%% Load data
data = pd.read_csv("./data/cleaned_heart_statlog.csv", sep=",")

data['sex'].replace({1: 'Male', 0: 'Female'}, inplace = True)
data['chest pain type'].replace({1: 'typical angina', 2: 'atypical angina', 3: 'non-anginal pain', 4: 'asymptomatic'}, inplace = True)
data['fasting blood sugar'].replace({1: 'bs > 120 mg/dl', 0: 'bs <= 120 mg/dl'}, inplace = True)
data['resting ecg'].replace({1: 'ST-T wave abnormality', 2: 'left ventricular hypertrophy', 0: 'normal'}, inplace = True)
data['exercise angina'].replace({1: 'exercise angina', 0: 'no exercise angina'}, inplace = True)
data['ST slope'].replace({1: 'upsloping', 2: 'flat', 3: 'downsloping'}, inplace = True)

data['target'].replace({1: 'heart disease', 0: 'Normal'}, inplace = True)

age_groups_descending = np.array(
    [ "80+", "75 - 79", "70 - 74",
     "65 - 69", "60 - 64", "55 - 59", "50 - 54", "45 - 49", "40 - 44", "35 - 39",
     "30 - 34", "25 - 29", "0 - 24"])

# Function to categorize age into age groups
def categorize_age(age):
    if age <= 24:
        return "0 - 24"
    elif age <= 29:
        return "25 - 29"
    elif age <= 34:
        return "30 - 34"
    elif age <= 39:
        return "35 - 39"
    elif age <= 44:
        return "40 - 44"
    elif age <= 49:
        return "45 - 49"
    elif age <= 54:
        return "50 - 54"
    elif age <= 59:
        return "55 - 59"
    elif age <= 64:
        return "60 - 64"
    elif age <= 69:
        return "65 - 69"
    elif age <= 74:
        return "70 - 74"
    elif age <= 79:
        return "75 - 79"
    else:
        return "80+"

data['age_group'] = data['age'].apply(categorize_age)

data = pd.get_dummies(data.drop(columns='age_group'))
data.drop(columns=["sex_Female", "chest pain type_asymptomatic", "target_Normal",
                              "fasting blood sugar_bs <= 120 mg/dl", "ST slope_upsloping",
                              "exercise angina_no exercise angina", "resting ecg_normal"], inplace = True)


X = data.drop(['target_heart disease'],axis=1)
y = data['target_heart disease']

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 42)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient

# Instantiate the logistic regression model
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)  # Train the model

# Predictions for evaluation
y_pred = logreg.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Log the model to MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient("http://127.0.0.1:5000")
experiment_name = "heart_disease_logreg"

try:
    client.create_experiment(experiment_name)
except Exception as e:
    pass  # Experiment might already exist

experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id):
    run_id = mlflow.active_run().info.run_id
    mlflow.sklearn.log_model(logreg, "model")
    mlflow.log_param("input", X_train.columns.to_list())
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    mlflow.log_metric("f1", f1_score(y_test, y_pred))

print(f"Model logged with run ID: {run_id}")

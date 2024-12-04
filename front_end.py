import streamlit as st
import pandas as pd
import pika
import requests
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, f1_score, recall_score
import matplotlib.pyplot as plt

#%% Simple interface

# RabbitMQ and API details
host = "localhost"
port = 5672
user = "guest"
password = "guest"
url = "http://localhost:8000"

# Upload CSV and Run ID input
upload = st.file_uploader("Upload Heart Disease Dataset (CSV format).")
run_id = st.text_input("Run ID")
if st.button("Load model") and (run_id is not None and run_id != ""):
    resp = requests.get(url + f"/model/{run_id}") 
    st.write("Model loaded: ", resp.content.decode("utf-8"))
else:
    st.write("Enter a valid Run ID and click the button to load the model.")

# Display the current loaded model
resp = requests.get(url + "/model/current") 
current_run_id = resp.content.decode("utf-8")
st.write(f"Current model loaded: {current_run_id}")

# If a CSV file is uploaded
if upload is not None:
    # Load data
    data = pd.read_csv(upload, sep=",")
    
    # Ensure the dataset includes the necessary features
    required_columns = ["age", "sex", "chest pain type", "resting bp s", "cholesterol", "fasting blood sugar", 
                        "resting ecg", "max heart rate", "exercise angina", "oldpeak", "ST slope"]
    if not all(col in data.columns for col in required_columns):
        st.error("The uploaded dataset does not contain all required columns.")
    else:
        # Send data to RabbitMQ
        connection = pika.BlockingConnection(pika.ConnectionParameters(host=host, port=port, credentials=pika.PlainCredentials(user, password)))
        channel = connection.channel()
        channel.queue_declare(queue="heart_disease", durable=True)
        channel.basic_publish(exchange='', routing_key="heart_disease", body=data.to_json().encode('utf-8'))
        connection.close()

        # Get predictions from the backend
        resp = requests.get(url + "/predict/heart_disease").json()

        # Process predictions
        predictions = pd.DataFrame.from_dict(json.loads(resp))
        
        # Scores display
        st.subheader("Model Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Precision", f"{precision_score(predictions['target'], predictions['y_pred']):.2f}")

        with col2:
            st.metric("Accuracy", f"{accuracy_score(predictions['target'], predictions['y_pred']):.2f}")

        with col3:
            st.metric("F1 Score", f"{f1_score(predictions['target'], predictions['y_pred']):.2f}")

        with col4:
            st.metric("Recall", f"{recall_score(predictions['target'], predictions['y_pred']):.2f}")

        # Confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(predictions["target"], predictions["y_pred"], ax=ax)
        st.pyplot(fig)
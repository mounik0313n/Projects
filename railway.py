import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
import requests

# Simulate Data Collection
def generate_acoustic_data(num_samples=1000):
    """Generates synthetic acoustic data with defect and non-defect conditions."""
    # Simulated features (e.g., frequency, amplitude, duration)
    frequency = np.random.uniform(100, 1000, num_samples)  # Hz
    amplitude = np.random.uniform(0.1, 2.0, num_samples)  # mV
    duration = np.random.uniform(0.05, 2.0, num_samples)  # seconds
    
    # Defect label (1 for defect, 0 for no defect)
    defect_label = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])  # 80% no defect, 20% defect
    
    # Combine the data into a DataFrame
    data = pd.DataFrame({
        'Frequency': frequency,
        'Amplitude': amplitude,
        'Duration': duration,
        'Defect': defect_label
    })
    
    return data

# Simulating Data Collection
data = generate_acoustic_data()

# Preprocess the acoustic data
def preprocess_data(data):
    """Preprocess the data by normalizing and filtering."""
    features = data[['Frequency', 'Amplitude', 'Duration']]
    
    # Standardizing features (normalization)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # In a real scenario, noise filtering could involve applying band-pass filters on raw signal data.
    
    return features_scaled, data['Defect']

# Preprocess the data
X, y = preprocess_data(data)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print results
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Visualize confusion matrix
plt.figure(figsize=(6, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['No Defect', 'Defect'])
plt.yticks(tick_marks, ['No Defect', 'Defect'])
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.tight_layout()
plt.show()

# Save defect detection results to CSV file
def save_results_to_csv(data, predictions, filename="defect_predictions.csv"):
    """Save the data and predictions to a CSV file."""
    results = data.copy()
    results['Predicted Defect'] = predictions
    results.to_csv(filename, index=False)
    print(f"Results saved to {filename}")
    
# Save the results to CSV
save_results_to_csv(data, y_pred)

# Simulate uploading data to a server
def upload_to_server(filename="defect_predictions.csv"):
    """Simulate uploading the CSV file to a server."""
    url = "https://example.com/upload"  # Example URL for uploading
    headers = {'Content-Type': 'application/json'}
    
    # Read the file contents
    with open(filename, 'r') as f:
        file_content = f.read()
    
    # Simulate sending data as a POST request
    data = {
        'filename': filename,
        'file_content': file_content
    }
    
    # Simulate a POST request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # Check the response
    if response.status_code == 200:
        print("Data uploaded successfully!")
    else:
        print(f"Failed to upload data. Status code: {response.status_code}")
    
# Simulate uploading the results to a server
upload_to_server()


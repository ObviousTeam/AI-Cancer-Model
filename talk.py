import joblib
import numpy as np

# Load the saved model
model = joblib.load('./ai.pkl')

# Define the input data
input_data = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871]
b_data = [11.31,19.04,71.8,394.1,0.08139,0.04701,0.03709,0.0223,0.1516,0.05667]

# Convert the input data to a 2D array, as the model expects a 2D input for a single instance
input_data = np.array([input_data])

# Make a prediction
prediction = model.predict(input_data)

# Output the result
print(f"Prediction for input: {prediction[0]}")


probabilities = model.predict_proba(input_data)

# Output the result
print(f"Confidence score for each class: {probabilities[0]}")


import time
import streamlit as st
import joblib
import numpy as np
import os

def run_model(input_data):
    # Load the saved model
    file_path = os.path.join(os.getcwd(), 'ai.pkl')
    print("File path:", file_path)
    print("File exists:", os.path.exists(file_path))


    model = joblib.load("S:\\pycharm\\CAC\\venv\\ai.pkl")

    # Convert the input data to a 2D array, as the model expects a 2D input for a single instance
    input_data = np.array([input_data])

    # Make a prediction
    prediction = model.predict(input_data)
    probabilities = model.predict_proba(input_data)

    prob = max(probabilities[0])

    return [prediction[0], prob]


st.title("Breast Cancer Prediction Model")

example_values = [17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871]

# Create input fields for all features with example values as placeholders
radius_mean = st.text_input("Radius Mean", placeholder=str(example_values[0]))
texture_mean = st.text_input("Texture Mean", placeholder=str(example_values[1]))
perimeter_mean = st.text_input("Perimeter Mean", placeholder=str(example_values[2]))
area_mean = st.text_input("Area Mean", placeholder=str(example_values[3]))
smoothness_mean = st.text_input("Smoothness Mean", placeholder=str(example_values[4]))
compactness_mean = st.text_input("Compactness Mean", placeholder=str(example_values[5]))
concavity_mean = st.text_input("Concavity Mean", placeholder=str(example_values[6]))
concave_points_mean = st.text_input("Concave Points Mean", placeholder=str(example_values[7]))
symmetry_mean = st.text_input("Symmetry Mean", placeholder=str(example_values[8]))
fractal_dimension_mean = st.text_input("Fractal Dimension Mean", placeholder=str(example_values[9]))

# Add a submit button
if st.button("Submit"):
    # Collect all inputs
    input_data = [
        float(radius_mean) if radius_mean else example_values[0],
        float(texture_mean) if texture_mean else example_values[1],
        float(perimeter_mean) if perimeter_mean else example_values[2],
        float(area_mean) if area_mean else example_values[3],
        float(smoothness_mean) if smoothness_mean else example_values[4],
        float(compactness_mean) if compactness_mean else example_values[5],
        float(concavity_mean) if concavity_mean else example_values[6],
        float(concave_points_mean) if concave_points_mean else example_values[7],
        float(symmetry_mean) if symmetry_mean else example_values[8],
        float(fractal_dimension_mean) if fractal_dimension_mean else example_values[9]
    ]

    with st.spinner('Processing...'):
        result = run_model(input_data)

    st.success("Prediction Complete!")

    # Display the results
    prediction = "Malignant" if result[0] == "M" else "Benign"
    st.write(f"Prediction: {prediction}")
    st.metric(label="Confidence", value=f"{result[1] * 100:.2f}%")

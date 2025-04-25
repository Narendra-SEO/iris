import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('iris_model.pkl')

st.title("🌸 Iris Flower Species Predictor")

st.write("Enter the measurements of the flower:")

# Input sliders
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.8)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.2)

# Prepare the features
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# Make prediction
prediction = model.predict(features)[0]
species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

# Show result
st.markdown(f"### 🌼 Predicted Species: **{species_map[prediction]}**")

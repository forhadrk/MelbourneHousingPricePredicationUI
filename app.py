import streamlit as st
import joblib
import os

# Path for the saved model
MODEL_PATH = "random_forest_model.pkl"

# Set the browser title and favicon
st.set_page_config(
    page_title="House Price Prediction App",  # Browser tab title
    page_icon="🏠",  # Favicon (emoji or path to an image file)
    layout="centered",  # Layout can be "centered" or "wide"
)

# Function to load the model
@st.cache_resource
def load_model():
    # Check if file exists
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Required model file is missing! Please upload the correct file.")
        return None
    # Load model
    model = joblib.load(MODEL_PATH)
    return model

# Load the model
model = load_model()

# Streamlit app title and description
st.title("🏠 House Price Prediction App")
st.subheader("🔍 Predict house prices based on key features.")
st.write("This app uses a machine learning model to predict house prices.")

# Input fields for features
st.sidebar.header("📋 Input Features")
rooms = st.sidebar.number_input("Number of Rooms", min_value=1, max_value=10, step=1, value=3)
bathrooms = st.sidebar.number_input("Number of Bathrooms", min_value=1, max_value=10, step=1, value=2)
car_spaces = st.sidebar.number_input("Number of Car Spaces", min_value=0, max_value=10, step=1, value=1)
landsize = st.sidebar.number_input("Landsize (sqm)", min_value=0, step=1, value=150)
building_area = st.sidebar.number_input("Building Area (sqm)", min_value=0, step=1, value=100)
year_built = st.sidebar.number_input("Year Built", min_value=1800, max_value=2025, step=1, value=1990)
distance = st.sidebar.number_input("Distance to City Center (km)", min_value=0.0, step=0.1, value=5.0)

# Prediction functionality
if st.button("Predict Price"):
    if model is None:
        st.error("❌ Model is not properly loaded. Please check your setup.")
    else:
        # Prepare input data
        input_data = [[rooms, bathrooms, car_spaces, landsize, building_area, year_built, distance]]

        # Make predictions
        predicted_price = model.predict(input_data)[0]

        # Display results
        st.success(f"🏡 Predicted House Price: **${predicted_price:,.2f}**")

# Footer with credits
st.markdown("---")
st.markdown("💡 Built with Streamlit by Forhad Hossain | House Price Prediction App")
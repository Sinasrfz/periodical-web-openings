import streamlit as st
import numpy as np
from joblib import load
from sklearn.preprocessing import MinMaxScaler
from PIL import Image
import requests
import os

# Function to download file from Google Drive
def download_from_drive(file_id, destination):
    """
    Downloads a file from Google Drive using its file ID.
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

# Google Drive file ID for the 'fy' model
file_id = "1L1AswG0Re3wLjMEjQDr7mmDVXYcY6z6L"  # Replace with your actual file ID
model_file = "SVR_best_model(fy).joblib"

# Download the model if it doesn't already exist
if not os.path.exists(model_file):
    st.info("Downloading the 'fy' model file. Please wait...")
    download_from_drive(file_id, model_file)

# Load the 'fy' model
try:
    fy_model = load(model_file)
except Exception as e:
    st.error(f"Failed to load the 'fy' model: {e}")

# Scaler initialization
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler.fit([[355.0, 221.26, 62.41, 34.67, 138.68], [960.0, 1354.64, 1502.83, 781.47, 1202.26]])

# Model loading function
def load_model(label_name):
    if label_name in ["d", "bf", "tf", "tw"]:
        model_name = "HistGB"
        model_path = f"{model_name}_best_model({label_name}).joblib"
        try:
            model = load(model_path)
        except FileNotFoundError:
            st.error(f"Model file {model_path} not found.")
            return None
        return model
    elif label_name == "fy":
        return fy_model  # Return the 'fy' model directly
    else:
        return None

# Load all models
models = {label: load_model(label) for label in ["d", "bf", "tf", "tw", "fy"]}

# Reference table for section properties
reference_sets = [
    [177.8, 101.2, 4.8, 7.9],
    [305.1, 101.6, 5.8, 7],
    [312.7, 102.4, 6.6, 10.8],
    [311, 125.3, 9, 14],
    [449.8, 152.4, 7.6, 10.9],
    [480.6, 196.7, 15.3, 26.3],
    [544.5, 211.9, 12.7, 21.3],
    [577.1, 320.2, 21.1, 37.6],
    [692.9, 255.8, 14.5, 23.7],
    [834.9, 291.7, 14, 18.8]
]

reference_names = [
    "UB 178x102x19",
    "UB 305x102x25",
    "UB 305x102x33",
    "UB 305x127x48",
    "UB 457x152x52",
    "UB 457x191x133",
    "UB 533x210x122",
    "UB 533x312x272",
    "UB 686x254x170",
    "UB 838x292x176"
]

reference_fy = [355, 460, 690, 960]

# Find the nearest set of section properties based on `d`
def find_nearest_set(d_value):
    reference_d = [row[0] for row in reference_sets]
    idx = (np.abs(np.array(reference_d) - d_value)).argmin()
    return reference_sets[idx], reference_names[idx]

# Find the nearest value for `fy`
def find_nearest_fy(fy_value):
    array = np.array(reference_fy)
    idx = (np.abs(array - fy_value)).argmin()
    return array[idx]

# Prediction function
def predict(features):
    features_array = np.array([features]).astype(np.float64)
    scaled_features = scaler.transform(features_array)  # Transform features using the scaler
    results = {label: float(models[label].predict(scaled_features)[0]) if models[label] else 'Model not loaded' for label in models}
    return results

# Streamlit UI
st.title("Prediction of Structural Features")

# Input form
st.sidebar.header("Enter Input Features")
wpb = st.sidebar.number_input("WPB [kN]", min_value=0.0, max_value=2000.0, step=1.0, value=400.0)
h = st.sidebar.number_input("h [mm]", min_value=0.0, max_value=2000.0, step=1.0, value=500.0)
s = st.sidebar.number_input("s [mm]", min_value=0.0, max_value=2000.0, step=1.0, value=100.0)
a0 = st.sidebar.number_input("a0 [mm]", min_value=0.0, max_value=2000.0, step=1.0, value=50.0)
h0 = st.sidebar.number_input("h0 [mm]", min_value=0.0, max_value=2000.0, step=1.0, value=150.0)

if st.sidebar.button("Predict"):
    features = [wpb, h, s, a0, h0]
    predictions = predict(features)

    # Process section properties (d, bf, tw, tf)
    if isinstance(predictions["d"], float):
        nearest_set, section_name = find_nearest_set(predictions["d"])
        st.subheader("Predicted Section Properties")
        st.write(f"**Section Name:** {section_name}")
        st.write(f"**Depth (d):** {nearest_set[0]:.2f} mm")
        st.write(f"**Flange Width (bf):** {nearest_set[1]:.2f} mm")
        st.write(f"**Web Thickness (tw):** {nearest_set[2]:.2f} mm")
        st.write(f"**Flange Thickness (tf):** {nearest_set[3]:.2f} mm")

    # Process `fy`
    if isinstance(predictions["fy"], float):
        nearest_fy = find_nearest_fy(predictions["fy"])
        st.write(f"**Yield Strength (fy):** {nearest_fy} MPa")

# Display the visualization
st.subheader("Visualization")
try:
    image = Image.open("Picture1.jpg")
    st.image(image, caption="Structural Features Visualization", use_container_width=True)  # Updated parameter
except Exception as e:
    st.error(f"Error loading image: {e}")

st.info("This web application was developed by SINA SARFARAZI.")

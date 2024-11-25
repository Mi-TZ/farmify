import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from PIL import Image
import io
import torch
from torchvision import transforms
from utils.model import ResNet9
from utils.disease import disease_dic
from utils.fertilizer import fertilizer_dic
import config

# Load models
disease_classes = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                   'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
                   'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
                   'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                   'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

disease_model_path = 'models/plant_disease_model.pth'
disease_model = ResNet9(3, len(disease_classes))
disease_model.load_state_dict(torch.load(disease_model_path, map_location=torch.device('cpu')))
disease_model.eval()

crop_recommendation_model_path = 'models/RandomForestModel.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


# Utility functions
def weather_fetch(city_name):
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    if x["cod"] != "404":
        y = x["main"]
        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None


def predict_image(img, model=disease_model):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction


# Streamlit UI
st.title("Farmify: Your Smart Agriculture Companion")
menu = ["Home", "Crop Recommendation", "Fertilizer Suggestion", "Disease Detection"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.subheader("Welcome to Farmify!")
    st.write("""
    Farmify is an intelligent platform to assist farmers and agricultural enthusiasts.
    - Recommend the best crop based on soil and weather conditions.
    - Suggest fertilizers tailored to your crop's nutrient needs.
    - Detect plant diseases from leaf images and provide solutions.
    """)

elif choice == "Crop Recommendation":
    st.subheader("Crop Recommendation")
    N = st.number_input("Nitrogen content (N)", min_value=0)
    P = st.number_input("Phosphorous content (P)", min_value=0)
    K = st.number_input("Potassium content (K)", min_value=0)
    ph = st.number_input("Soil pH value", min_value=0.0, max_value=14.0, step=0.1)
    rainfall = st.number_input("Rainfall (in mm)", min_value=0.0)
    city = st.text_input("Enter city name for weather data")

    if st.button("Recommend Crop"):
        if weather_fetch(city):
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            prediction = crop_recommendation_model.predict(data)
            st.success(f"The recommended crop for your conditions is: {prediction[0]}")
        else:
            st.error("Could not fetch weather data. Please try again.")

elif choice == "Fertilizer Suggestion":
    st.subheader("Fertilizer Suggestion")

    # Load crop list from the CSV file
    df = pd.read_csv('Data/fertilizer.csv')
    crop_list = df['Crop'].unique().tolist()  # Get unique crops from the data

    crop_name = st.selectbox("Select Crop", crop_list)  # Dropdown for crop names
    N = st.number_input("Nitrogen content (N)", min_value=0)
    P = st.number_input("Phosphorous content (P)", min_value=0)
    K = st.number_input("Potassium content (K)", min_value=0)

    if st.button("Suggest Fertilizer"):
        # Fetch nutrient requirements for the selected crop
        crop_data = df[df['Crop'] == crop_name].iloc[0]
        nr, pr, kr = crop_data['N'], crop_data['P'], crop_data['K']

        n = nr - N
        p = pr - P
        k = kr - K

        # Determine the key for fertilizer suggestion
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_value = temp[max(temp.keys())]
        if max_value == "N":
            key = 'NHigh' if n < 0 else 'Nlow'
        elif max_value == "P":
            key = 'PHigh' if p < 0 else 'Plow'
        else:
            key = 'KHigh' if k < 0 else 'Klow'

        # Get fertilizer recommendation
        recommendation = fertilizer_dic.get(key, "No recommendation found")
        st.html(f"We recommend using: {recommendation}")


elif choice == "Disease Detection":
    st.subheader("Plant Disease Detection")
    file = st.file_uploader("Upload a leaf image", type=["jpg", "png", "jpeg"])
    if file is not None:
        img = file.read()
        try:
            prediction = predict_image(img)
            st.write(f"Detected Disease: {prediction}")
            st.html(disease_dic[prediction])
        except Exception as e:
            st.error("Could not process the image. Please try again.")

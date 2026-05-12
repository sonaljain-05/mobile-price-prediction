import numpy as np
import streamlit as st
import joblib
from PIL import Image
import pandas as pd


final_model = joblib.load("price_prediction.pkl")
scaler = joblib.load("scaler.pkl")


st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(to right,#EBFCFF, #63A1FF);
    }
    </style>
    """,
    unsafe_allow_html=True
)








st.title("MOBILE PRICE PREDICTION APP")



st.write("This app predicts mobile price based on specifications.")
st.subheader("Please enter the phone specifications below")

img = Image.open("mobile.png")
img=img.resize((2000, 1000))
st.image(img)

st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

st.header("Enter Smartphone Features")
st.markdown("<br><br>", unsafe_allow_html=True)


img2 = Image.open("ram.jpg")


img2 = img2.resize((100, 100))

col1, col2 = st.columns([1, 3])

with col1:
    st.image(img2)

with col2:
    st.header("performance details")






ram = st.selectbox("Enter the RAM of the mobile (in GB):", options=list(range(1, 17)))



internal_mem= st.slider("Enter the internal memory of the mobile (in GB):", min_value=1, max_value=128, step=1)


cpu_freq = st.selectbox("Enter the CPU frequency (in GHz):", options=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
cpu_core = st.selectbox(
    "Enter the number of CPU cores:",
    [1, 2, 4, 6, 8, 12, 16]
)
ppi = st.slider("Enter the pixel density (PPI) of the display:", min_value=100, max_value=800, step=1)


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

img_camera = Image.open("camera.jpg")  # camera icon/image

img_camera = img_camera.resize((100, 100))

col1, col2 = st.columns([1, 3])

with col1:
    st.image(img_camera)

with col2:
    st.header("camera and display details")
    





rear_camera = st.selectbox("Enter the rear camera resolution (in MP):", options=[8, 12, 13, 16, 48, 50, 64, 108, 200])


front_camera = st.selectbox("Enter the front camera resolution (in MP):", options=[5, 8, 10, 12, 16, 20, 32])



resolution = st.slider("Enter the Screen Size (in inches):", min_value=1.4, max_value=12.2, value=5.2, step=0.1)

        

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)

img_battery = Image.open("battery.webp")  # ya koi battery image
img_battery = img_battery.resize((100, 100))

col1, col2 = st.columns([1, 3])

with col1:
    st.image(img_battery)

with col2:
    st.header("build and battery details")


battery= st.slider("Enter the battery capacity (in mAh):", min_value=800, max_value=9500, step=100)


weight = st.slider("Enter the weight of the mobile (in grams):", min_value=100, max_value=200, step=1)

thickness = st.slider("Enter the thickness of the mobile (in mm):", min_value=5.0, max_value=20.0, step=0.1)

sales = st.slider("Enter the number of sales (in millions):", min_value=0.0, max_value=100.0, step=0.1)












st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)












st.write("### 📋 Input Preview Table  📋")



st.markdown("<br><br>", unsafe_allow_html=True)

preview_dict = {
        "Sales (M)": sales,
        "Weight (g)": weight,
        "Screen (Inch)": resolution,
        "PPI": ppi,
        "CPU Cores": cpu_core,
        "CPU GHz": cpu_freq,
        "Internal Memory": internal_mem,
        "RAM": ram,
        "Rear Camera": rear_camera,
        "Front Camera": front_camera,
        "Battery": battery,
        "Thickness": thickness
    }

preview_df = pd.DataFrame([preview_dict])


st.dataframe(preview_df, use_container_width=True)



if st.button(" PREDICT MOBILE PRICE"):
    
    input_dict = {
        "Sale": np.log1p(sales),
        "weight": np.log1p(weight),
        "resolution": np.log1p(resolution),
        "ppi": ppi,
        "cpu_core": cpu_core,
        "cpu_freq": cpu_freq,
        "internal_mem": np.log1p(internal_mem),
        "ram": ram,
        "RearCam": rear_camera,
        "Front_Cam": np.log1p(front_camera),
        "battery": np.log1p(battery),
        "thickness": np.log1p(thickness)
    }

    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = final_model.predict(input_scaled)

    st.success("Specifications Analyzed!")

    st.metric(
        label="Estimated Price",
        value=f"{prediction[0]:,.2f}"
    )

    st.caption("*Based on trained machine learning model")




  

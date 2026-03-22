import streamlit as st
import pandas as pd
import joblib
import numpy as np

# โหลดโมเดลที่เทรนไว้
model = joblib.load('weather_model.pkl')

# ส่วนของ UI [cite: 37]
st.set_page_config(page_title="Weather Predictor", page_icon="🌦️")
st.title("🌡️ Apparent Temperature Predictor")
st.write("โปรแกรมทำนายอุณหภูมิที่ 'รู้สึกจริง' จากปัจจัยสภาพอากาศ")

# ส่วนรับข้อมูล (Input Validation) [cite: 37]
with st.sidebar:
    st.header("ใส่ข้อมูลสภาพอากาศ")
    temp = st.number_input("อุณหภูมิจริง (Celsius)", value=15.0)
    humidity = st.slider("ความชื้น (0.0 - 1.0)", 0.0, 1.0, 0.5)
    wind = st.number_input("ความเร็วลม (km/h)", min_value=0.0, value=10.0)
    vis = st.number_input("ทัศนวิสัย (km)", min_value=0.0, max_value=20.0, value=10.0)
    press = st.number_input("ความกดอากาศ (mbar)", min_value=900.0, max_value=1100.0, value=1010.0)

# ทำนายผล
input_data = pd.DataFrame([[temp, humidity, wind, vis, press]], 
                          columns=['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)'])

if st.button("ทำนายผล"):
    prediction = model.predict(input_data)[0]
    
    # แสดงผลลัพธ์ [cite: 37]
    st.metric(label="อุณหภูมิที่รู้สึกจริง (Apparent Temperature)", value=f"{prediction:.2f} °C")
    
    if prediction > 30:
        st.warning("อากาศค่อนข้างร้อน ควรดื่มน้ำมากๆ")
    elif prediction < 10:
        st.info("อากาศค่อนข้างหนาว ควรเตรียมเสื้อกันหนาว")

st.divider()
st.caption("จัดทำโดย: [ชื่อของคุณ] | ข้อมูล: Szeged Weather Dataset")

print("Done")
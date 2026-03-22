import streamlit as st
import pandas as pd
import joblib

# 1. ตั้งค่าหน้าเว็บ (ทำให้ออกมาดูดีบนมือถือและคอม)
st.set_page_config(page_title="Weather Predictor Pro", page_icon="🌡️", layout="wide")

# 2. โหลดโมเดล (ใช้ Path หน้าแรกตามที่เราตกลงกัน)
@st.cache_resource
def load_model():
    return joblib.load('weather_model.pkl')

model = load_model()

# 3. ส่วนของ Sidebar (แถบด้านข้างสำหรับกรอกข้อมูล)
st.sidebar.header("📥 กรอกข้อมูลสภาพอากาศ")
st.sidebar.markdown("ปรับค่าด้านล่างเพื่อพยากรณ์อุณหภูมิที่รู้สึกจริง")

temp = st.sidebar.number_input("อุณหภูมิจริง (Celsius)", value=30.0, step=0.1)
humidity = st.sidebar.slider("ความชื้นสัมพัทธ์ (0.0 - 1.0)", 0.0, 1.0, 0.6)
wind_speed = st.sidebar.number_input("ความเร็วลม (km/h)", value=10.0, step=0.5)
visibility = st.sidebar.number_input("ทัศนวิสัย (km)", value=10.0, step=0.5)
pressure = st.sidebar.number_input("ความกดอากาศ (mbar)", value=1010.0, step=0.1)

# 4. หน้าหลักของแอป
st.title("🌡️ Apparent Temperature Predictor")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 ข้อมูลที่คุณระบุ")
    input_data = pd.DataFrame([[temp, humidity, wind_speed, visibility, pressure]], 
                              columns=['Temperature (C)', 'Humidity', 'Wind Speed (km/h)', 'Visibility (km)', 'Pressure (millibars)'])
    st.table(input_data)

with col2:
    st.subheader("🚀 ผลการวิเคราะห์ด้วย AI")
    if st.button("คลิกเพื่อทำนายผล"):
        # ทำนายผล
        prediction = model.predict(input_data)[0]
        diff = prediction - temp
        
        # แสดงผลแบบ Metric Card
        st.metric(label="อุณหภูมิที่รู้สึกจริง (Apparent Temp)", 
                  value=f"{prediction:.2f} °C", 
                  delta=f"{diff:.2f} °C จากอุณหภูมิจริง")

        # 5. การแปลผลและคำแนะนำ (Logic อัปเกรดคะแนน)
        if prediction >= 35:
            st.error("🔴 **อันตราย:** อากาศร้อนจัด เสี่ยงโรคลมแดด (Heatstroke) หลีกเลี่ยงกิจกรรมกลางแจ้ง")
        elif prediction >= 28:
            st.warning("🟡 **เตือน:** อากาศค่อนข้างร้อน ควรดื่มน้ำให้เพียงพอและอยู่ในที่ร่ม")
        elif prediction >= 15:
            st.success("🟢 **ปกติ:** อากาศกำลังสบาย เหมาะแก่การทำกิจกรรมทั่วไป")
        else:
            st.info("🔵 **หนาว:** อากาศค่อนข้างเย็น ควรเตรียมเสื้อกันหนาว")

st.markdown("---")
st.caption("จัดทำโดย: กิตติพัฒน์ แน่นอุดร | ข้อมูลอ้างอิง: Szeged Weather Dataset")

# 🌡️ Apparent Temperature Predictor

> ระบบพยากรณ์อุณหภูมิที่รู้สึกจริง (Apparent Temperature) ด้วย Machine Learning  
> จัดทำโดย **กิตติพัฒน์ แน่นอุดร** | ข้อมูลอ้างอิง: Szeged Weather Dataset

---

## 📌 ภาพรวมโปรเจค

โปรเจคนี้สร้างโมเดล Machine Learning เพื่อทำนาย **อุณหภูมิที่ร่างกายรู้สึกจริง (Apparent Temperature)** จากข้อมูลสภาพอากาศ ซึ่งมีความสำคัญในทางปฏิบัติ เนื่องจากอุณหภูมิที่อ่านได้จากเทอร์โมมิเตอร์อาจแตกต่างจากที่ร่างกายรับรู้จริงอย่างมาก เนื่องจากปัจจัยต่างๆ เช่น ความชื้น ความเร็วลม และทัศนวิสัย

**ประเภทปัญหา:** Regression  
**เป้าหมาย:** ทำนาย `Apparent Temperature (C)` จาก features ด้านสภาพอากาศ

---

## 🗂️ โครงสร้างไฟล์

```
weather-predictor/
├── data/
│   └── weatherHistory.csv      # Dataset หลัก (Szeged Weather, ~96,000 rows)
├── app.py                      # Streamlit Web Application
├── train_model.py              # Script สำหรับ Train โมเดล
├── weather_model.pkl           # โมเดลที่ Train แล้ว (สร้างหลัง run train_model.py)
├── requirements.txt            # Python dependencies
└── README.md                   # ไฟล์นี้
```

---

## 📊 Dataset

| รายละเอียด | ข้อมูล |
|---|---|
| **ชื่อ Dataset** | Szeged Weather Dataset |
| **แหล่งที่มา** | Kaggle / Dark Sky API |
| **จำนวนแถว** | ~96,000 rows |
| **ช่วงเวลา** | 2006–2016 |

### Features ที่ใช้

| Feature | คำอธิบาย |
|---|---|
| `Temperature (C)` | อุณหภูมิจริงที่วัดได้ (°C) |
| `Humidity` | ความชื้นสัมพัทธ์ (0.0–1.0) |
| `Wind Speed (km/h)` | ความเร็วลม (กม./ชม.) |
| `Visibility (km)` | ระยะทัศนวิสัย (กม.) |
| `Pressure (millibars)` | ความกดอากาศ (mbar) |

**Target:** `Apparent Temperature (C)` — อุณหภูมิที่ร่างกายรู้สึกจริง

---

## ⚙️ การเตรียมข้อมูล

- **Outlier Treatment:** ค่า `Pressure (millibars) = 0` ถูกแทนด้วยค่า Median ของข้อมูลที่ถูกต้อง
- **Pipeline:** รวม `StandardScaler` และโมเดลเข้าด้วยกันเพื่อป้องกัน Data Leakage
- **Train/Test Split:** 80% / 20% (random_state=42)

---

## 🤖 โมเดลและผลลัพธ์

**Algorithm:** Random Forest Regressor  
**Hyperparameter Tuning:** GridSearchCV (cv=5)

| Hyperparameter | ค่าที่ทดสอบ |
|---|---|
| `n_estimators` | 50, 100 |
| `max_depth` | None, 10 |

### ผลการประเมิน

| Metric | ผลลัพธ์ |
|---|---|
| **MAE** | ~0.5 °C |
| **R² Score** | ~0.99 |

> R² ใกล้ 1.0 แสดงว่าโมเดลสามารถอธิบายความแปรปรวนของอุณหภูมิที่รู้สึกได้เกือบทั้งหมด

---

## 🚀 การติดตั้งและใช้งาน

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train โมเดล

```bash
python train_model.py
```

จะสร้างไฟล์ `weather_model.pkl` ในโฟลเดอร์เดียวกัน

### 3. รัน Web App

```bash
streamlit run app.py
```

เปิดเบราว์เซอร์ที่ `http://localhost:8501`

---

## 🌐 Web Application

แอปพลิเคชัน Streamlit ให้ผู้ใช้สามารถ:
- กรอกข้อมูลสภาพอากาศผ่าน Sidebar
- ดูผลทำนายอุณหภูมิที่รู้สึกพร้อม Delta จากอุณหภูมิจริง
- อ่านคำแนะนำด้านสุขภาพตามระดับความร้อน

| ระดับ | อุณหภูมิที่รู้สึก | คำแนะนำ |
|---|---|---|
| 🔴 อันตราย | ≥ 35°C | หลีกเลี่ยงกิจกรรมกลางแจ้ง |
| 🟡 เตือน | 28–34°C | ดื่มน้ำให้เพียงพอ |
| 🟢 ปกติ | 15–27°C | เหมาะแก่กิจกรรมทั่วไป |
| 🔵 หนาว | < 15°C | เตรียมเสื้อกันหนาว |

**Live Demo:** [Streamlit Cloud URL]

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
joblib
```

---

## 👤 ผู้จัดทำ

**กิตติพัฒน์ แน่นอุดร**  
ข้อมูลอ้างอิง: [Szeged Weather Dataset on Kaggle](https://www.kaggle.com/budincsevity/szeged-weather)

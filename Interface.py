import streamlit as st
import pandas as pd
import pickle  # ใช้ pickle แทน joblib

# โหลดโมเดลจากไฟล์
with open("meal_match_model.pkl", "rb") as f:
    model = pickle.load(f)

# ข้อมูลตัวอย่างร้านอาหาร
data = {
    "name": ["ร้าน A", "ร้าน B", "ร้าน C", "ร้าน D", "ร้าน E", "ร้าน F"],
    "location": ["ประตู 1", "ประตู 1", "ประตู 3", "ประตู 4", "ประตู 1", "ประตู 2"],
    "choice": ["อาหารตามสั่ง", "อาหารตามสั่ง", "อาหารจานเดียว", "ปิ้งย่าง", "อาหารเกาหลี", "อาหารญี่ปุ่น"],
    "budget": ["50 - 100", "50 - 100", "50 - 100", "200+", "100 - 200", "50 - 100"],
    "time": ["กลางวัน", "กลางวัน", "เช้า", "กลางวัน", "เย็น", "เช้า"]
}
df = pd.DataFrame(data)

# ฟังก์ชันกรองร้าน
def filter_restaurants(location, food_type, price_range, time_of_day):
    return df[
        (df['location'] == location) &
        (df['choice'] == food_type) &
        (df['budget'] == price_range) &
        (df['time'] == time_of_day)
    ]['name'].tolist()

# Streamlit UI
st.set_page_config(page_title="MealMatch 🍽️", layout="centered")
st.title("🍽️ MealMatch - มื้อไหนดี?")

# === ตั้งค่าผู้ใช้ ===
user_location = st.selectbox("📍 บริเวณที่ต้องการจะไป", ["ประตู 1", "ประตู 2", "ประตู 3", "ประตู 4"])
user_choice = st.selectbox("🍱 เลือกประเภทอาหาร", ["อาหารตามสั่ง", "อาหารอีสาน", "อาหารจานเดียว", "ปิ้งย่าง", "อาหารเกาหลี", "อาหารญี่ปุ่น"])
user_budget = st.radio("💸 งบประมาณต่อมื้อ (บาท)", ["ไม่เกิน 50", "50 - 100", "100 - 200", "200+"])
user_time = st.selectbox("⏰ เวลาที่มักออกไปกิน", ["เช้า", "กลางวัน", "เย็น"])

# ฟอร์มสำหรับ submit
if st.button("🔍 ค้นหาร้านอาหาร"):
    user_input = pd.DataFrame([{
        "location": user_location,
        "choice": user_choice,
        "budget": user_budget,
        "time": user_time
    }])

    # One-hot encoding ข้อมูล
    user_input_encoded = pd.get_dummies(user_input)

    # เติมค่าที่ขาดหาย
    for col in df.drop("name", axis=1).columns:
        if col not in user_input_encoded.columns:
            user_input_encoded[col] = 0

    # จัดเรียง columns ให้ตรงกัน
    user_input_encoded = user_input_encoded[df.drop("name", axis=1).columns]

    # ทำนายผล
    prediction = model.predict(user_input_encoded)[0]

    if prediction == 1:
        matched_restaurants = filter_restaurants(user_location, user_choice, user_budget, user_time)
        if matched_restaurants:
            st.success("ร้านที่ตรงกับคุณมีดังนี้ 🍜")
            st.write(matched_restaurants)
        else:
            st.error("ไม่พบร้านอาหารที่ตรงกับตัวเลือกของคุณ 😥")
    else:
        st.warning("อาจจะไม่ใช่ร้านที่ตรงใจ ลองเปลี่ยนตัวเลือกดูนะ 😊")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle  # ใช้ pickle แทน joblib

# ข้อมูลตัวอย่าง (ใช้ตามที่คุณต้องการ)
data = {
    "location": ["ประตู 1", "ประตู 2", "ประตู 3", "ประตู 4", "ประตู 1", "ประตู 2"],
    "choice": ["อาหารตามสั่ง", "อาหารอีสาน", "อาหารจานเดียว", "ปิ้งย่าง", "อาหารญี่ปุ่น", "อาหารเกาหลี"],
    "budget": ["50 - 100", "100 - 200", "50 - 100", "200+", "100 - 200", "50 - 100"],
    "time": ["กลางวัน", "เย็น", "เช้า", "กลางวัน", "เย็น", "กลางวัน"],
    "selected_store": [1, 0, 1, 0, 1, 0]  # 1 คือร้านที่เลือก, 0 คือไม่ได้เลือก
}

df = pd.DataFrame(data)

# แปลงข้อมูลเป็น One-Hot Encoding
df_encoded = pd.get_dummies(df)

# แยกข้อมูลออกเป็น X (features) และ y (target)
X = df_encoded.drop("selected_store", axis=1)
y = df_encoded["selected_store"]

# แบ่งข้อมูลเป็น training และ test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# สร้างและฝึกโมเดล
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# บันทึกโมเดลด้วย pickle
with open("meal_match_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("โมเดลถูกฝึกและบันทึกสำเร็จ!")

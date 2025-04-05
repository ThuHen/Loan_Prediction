import streamlit as st
import numpy as np
import joblib

# Tải mô hình và scaler đã huấn luyện
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("📊 Dự đoán khả năng trả nợ")

# Giao diện nhập liệu với giải thích giá trị
collateral = st.selectbox(
    "Tài sản thế chấp:",
    [0, 1],
    format_func=lambda x: "0 - Có tài sản thế chấp" if x == 0 else "1 - Không có tài sản thế chấp"
)

income_proof = st.selectbox(
    "Chứng minh thu nhập:",
    [0, 1],
    format_func=lambda x: "0 - Có giấy tờ chứng minh thu nhập" if x == 0 else "1 - Không có giấy tờ chứng minh thu nhập"
)

marital_status = st.selectbox(
    "Tình trạng hôn nhân:",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 - Có gia đình",
        2: "2 - Độc thân",
        3: "3 - Ly hôn",
        4: "4 - Góa"
    }[x]
)

income = st.number_input("Thu nhập hàng tháng (VND):", min_value=0, step=500000)

education = st.selectbox(
    "Trình độ học vấn:",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 - Tiến sĩ",
        2: "2 - Thạc sĩ",
        3: "3 - Đại học",
        4: "4 - Cấp ba"
    }[x]
)

house_status = st.selectbox(
    "Tình trạng sở hữu nhà:",
    [0, 1],
    format_func=lambda x: "0 - Đã sở hữu nhà" if x == 0 else "1 - Chưa sở hữu nhà"
)

# Gộp dữ liệu đầu vào
input_data = np.array([[collateral, income_proof, marital_status, income, education, house_status]])

# Chuẩn hóa dữ liệu
input_scaled = scaler.transform(input_data)

# Dự đoán
if st.button("📌 Dự đoán"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 0:
        st.success("✅ Khách hàng có khả năng **trả nợ đúng hạn**.")
    else:
        st.error("⚠️ Khách hàng **có nguy cơ trả nợ trễ hạn**.")

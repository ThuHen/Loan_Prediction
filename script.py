import streamlit as st
import numpy as np
import joblib

# Tải mô hình và scaler đã huấn luyện
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Dự đoán khả năng **phê duyệt khoản vay**")

# Giao diện nhập liệu với giải thích giá trị
income_proof = st.selectbox(
    "Chứng minh thu nhập:",
    [0, 1],
    format_func=lambda x: "0 - Có giấy tờ chứng minh thu nhập" if x == 0 else "1 - Không có giấy tờ chứng minh thu nhập"
)

collateral = st.selectbox(
    "Tài sản thế chấp:",
    [0, 1],
    format_func=lambda x: "0 - Có tài sản thế chấp" if x == 0 else "1 - Không có tài sản thế chấp"
)

income = st.number_input("Thu nhập hàng tháng (VND):", min_value=0, step=500000, value=20000000)

marital_status = st.selectbox(
    "Tình trạng hôn nhân:",
    [1, 2, 3, 4],
    format_func=lambda x: {
        1: "1 - Có gia đình",
        2: "2 - Độc thân",
        3: "3 - Ly hôn",
        4: "4 - Góa"
    }.get(x, "Không xác định")
)
# Hóa đơn điện
electric_bill = st.number_input("Hóa đơn tiền điện (VND):", min_value=0, step=100000, value=1000000)
# Gộp dữ liệu đầu vào
input_data = np.array([
    income_proof,
    collateral,
    income,
    marital_status,
    electric_bill
]).reshape(1, -1)

# Chuẩn hóa dữ liệu
input_scaled = scaler.transform(input_data)

# Dự đoán
if st.button("📌 Dự đoán kết quả phê duyệt"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 0:
        st.success("✅ Khoản vay **có thể được phê duyệt** – khách hàng đủ điều kiện.")
    else:
        st.error("❌ Khoản vay **có thể bị từ chối** – khách hàng chưa đủ điều kiện.")
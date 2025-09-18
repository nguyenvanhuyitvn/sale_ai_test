import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =============================
# 1. Cấu hình ứng dụng
# =============================
st.set_page_config(page_title="Dự đoán Doanh thu với Linear Regression", layout="wide")
st.title("📊 Phân tích & Dự đoán Doanh thu (Linear Regression)")

# =============================
# 2. Upload file Excel
# =============================
uploaded_file = st.file_uploader("⬆️ Tải file Excel dữ liệu bán hàng", type=["xlsx"])

if uploaded_file is not None:
    # ---------------------------
    # 3. Đọc dữ liệu
    # ---------------------------
    df = pd.read_excel(uploaded_file)
    df["Ngày"] = pd.to_datetime(df["Ngày"])
    st.success("✅ Đã tải dữ liệu thành công!")

    # ---------------------------
    # 4. Feature Engineering
    # ---------------------------
    df["PQ"] = df["Giá"] * df["Số lượng"]
    df["Disc_PQ"] = (df["Giảm giá (%)"]/100) * df["PQ"]
    df["Tháng"] = df["Ngày"].dt.month
    df["Ngày_trong_tuần"] = df["Ngày"].dt.dayofweek

    # ---------------------------
    # 5. Thống kê nhanh
    # ---------------------------
    st.subheader("📌 Tóm tắt kinh doanh")
    st.metric("Tổng doanh thu", f"{df['Doanh thu'].sum():,.0f} VND")
    st.metric("Doanh thu TB/ngày", f"{df.groupby('Ngày')['Doanh thu'].sum().mean():,.0f} VND")

    # ---------------------------
    # 6. Chuẩn bị dữ liệu cho ML
    # ---------------------------
    numeric_features = ["PQ", "Disc_PQ", "Tháng", "Ngày_trong_tuần"]
    categorical_features = ["Danh mục", "Nhà cung cấp", "Khu vực", "Khách hàng", "Phương thức thanh toán", "Sản phẩm"]

    X = df[numeric_features + categorical_features]
    y = df["Doanh thu"]

    preprocessor = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("lr", LinearRegression())
    ])

    # ---------------------------
    # 7. Train/Test split + Fit
    # ---------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ---------------------------
    # 8. Biểu đồ Actual vs Predicted
    # ---------------------------
    st.subheader("📈 So sánh Doanh thu thực tế vs Dự đoán")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(y_test, y_pred, alpha=0.6, color="blue", label="Dự đoán")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", label="Đường y=x")
    ax.set_xlabel("Thực tế")
    ax.set_ylabel("Dự đoán")
    ax.set_title("Actual vs Predicted (Linear Regression)")
    ax.legend()
    st.pyplot(fig)

    # ---------------------------
    # 9. Form nhập liệu dự đoán mới
    # ---------------------------
    st.subheader("📝 Dự đoán Doanh thu cho đơn hàng mới")

    with st.form("predict_form"):
        soluong = st.number_input("Số lượng", min_value=1, value=1)
        gia = st.number_input("Giá (VND)", min_value=1000, value=1000000, step=1000)
        giamgia = st.slider("Giảm giá (%)", 0, 50, 0)
        thang = st.selectbox("Tháng", list(range(1,13)))
        ngay_trong_tuan = st.selectbox("Ngày trong tuần (0=Thứ 2)", list(range(7)))
        danhmuc = st.selectbox("Danh mục", df["Danh mục"].unique())
        nhacc = st.selectbox("Nhà cung cấp", df["Nhà cung cấp"].unique())
        khuvuc = st.selectbox("Khu vực", df["Khu vực"].unique())
        khachhang = st.selectbox("Khách hàng", df["Khách hàng"].unique())
        pt_thanhtoan = st.selectbox("Phương thức thanh toán", df["Phương thức thanh toán"].unique())
        sanpham = st.selectbox("Sản phẩm", df["Sản phẩm"].unique())

        submitted = st.form_submit_button("Dự đoán Doanh thu")

    if submitted:
        input_df = pd.DataFrame([{
            "PQ": gia * soluong,
            "Disc_PQ": (giamgia/100) * gia * soluong,
            "Tháng": thang,
            "Ngày_trong_tuần": ngay_trong_tuan,
            "Danh mục": danhmuc,
            "Nhà cung cấp": nhacc,
            "Khu vực": khuvuc,
            "Khách hàng": khachhang,
            "Phương thức thanh toán": pt_thanhtoan,
            "Sản phẩm": sanpham
        }])

        prediction = model.predict(input_df)[0]
        st.success(f"📌 Linear Regression dự đoán Doanh thu: {prediction:,.0f} VND")

    # ---------------------------
    # 10. Dự đoán trực quan Top sản phẩm (cải tiến)
    # ---------------------------
    st.subheader("📊 Dự đoán Doanh thu cho Top sản phẩm (theo giá trị riêng)")

    top_products = df["Sản phẩm"].value_counts().head(5).index

    most_region = df["Khu vực"].mode()[0]
    most_customer = df["Khách hàng"].mode()[0]
    most_supplier = df["Nhà cung cấp"].mode()[0]
    most_category = df["Danh mục"].mode()[0]
    most_payment = df["Phương thức thanh toán"].mode()[0]

    pred_results = []
    for prod in top_products:
        avg_price = df[df["Sản phẩm"] == prod]["Giá"].mean()
        avg_qty = df[df["Sản phẩm"] == prod]["Số lượng"].mean()
        avg_discount = df[df["Sản phẩm"] == prod]["Giảm giá (%)"].mean()

        input_df = pd.DataFrame([{
            "PQ": avg_price * avg_qty,
            "Disc_PQ": (avg_discount/100) * avg_price * avg_qty,
            "Tháng": 6,
            "Ngày_trong_tuần": 2,
            "Danh mục": most_category,
            "Nhà cung cấp": most_supplier,
            "Khu vực": most_region,
            "Khách hàng": most_customer,
            "Phương thức thanh toán": most_payment,
            "Sản phẩm": prod
        }])

        pred = model.predict(input_df)[0]
        pred_results.append((prod, pred))

    pred_df = pd.DataFrame(pred_results, columns=["Sản phẩm", "Doanh thu dự đoán"])

    # Bar chart
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Doanh thu dự đoán", y="Sản phẩm", data=pred_df, ax=ax)
    ax.set_title("Dự đoán Doanh thu cho Top sản phẩm (theo giá trị riêng)")
    st.pyplot(fig)

    # Line chart
    fig, ax = plt.subplots(figsize=(8,5))
    sns.lineplot(x="Sản phẩm", y="Doanh thu dự đoán", data=pred_df, marker="o", ax=ax)
    ax.set_title("Xu hướng doanh thu dự đoán theo sản phẩm")
    st.pyplot(fig)

    # Pie chart
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(pred_df["Doanh thu dự đoán"], labels=pred_df["Sản phẩm"], autopct="%1.1f%%", startangle=140)
    ax.set_title("Tỷ trọng doanh thu dự đoán giữa sản phẩm")
    st.pyplot(fig)

    # Scatter plot: giá trung bình vs doanh thu dự đoán
    avg_prices = df.groupby("Sản phẩm")["Giá"].mean().reset_index()
    scatter_df = pred_df.merge(avg_prices, on="Sản phẩm", how="left")

    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(scatter_df["Giá"], scatter_df["Doanh thu dự đoán"], s=100, alpha=0.7)
    for i, row in scatter_df.iterrows():
        ax.text(row["Giá"], row["Doanh thu dự đoán"], row["Sản phẩm"], fontsize=8)
    ax.set_xlabel("Giá trung bình (VND)")
    ax.set_ylabel("Doanh thu dự đoán (VND)")
    ax.set_title("Giá trung bình vs Doanh thu dự đoán")
    st.pyplot(fig)

    # Area chart
    pred_df_sorted = pred_df.sort_values("Doanh thu dự đoán", ascending=False)
    pred_df_sorted["Cộng dồn"] = pred_df_sorted["Doanh thu dự đoán"].cumsum()

    fig, ax = plt.subplots(figsize=(8,5))
    ax.fill_between(pred_df_sorted["Sản phẩm"], pred_df_sorted["Cộng dồn"], color="skyblue", alpha=0.4)
    ax.plot(pred_df_sorted["Sản phẩm"], pred_df_sorted["Cộng dồn"], marker="o", color="blue")
    ax.set_title("Doanh thu dự đoán tích lũy theo sản phẩm")
    ax.set_ylabel("Tích lũy Doanh thu dự đoán (VND)")
    st.pyplot(fig)

else:
    st.info("⬆️ Vui lòng tải file Excel để bắt đầu phân tích & dự đoán.")

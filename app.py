import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
st.title("Phân tích dữ liệu bán hàng (EDA) & Minh họa Mean vs median")
#--- 0. Upload file Excel ---
st.sidebar.header("1. Tải file dữ liệu")
uploaded_file = st.sidebar.file_uploader("Chọn file Excel", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # st.write("📌 Kiểu dữ liệu:")
    # st.write(df.dtypes)

    # for col in df.columns:
    #     unique_types = df[col].dropna().map(type).unique()
    #     st.write(f"👉 Cột {col}: {unique_types}")

    # df_fixed = df.convert_dtypes()  # Pandas tự đoán kiểu tốt nhất
    #df_fixed = df_fixed.astype(str) # Nếu vẫn lỗi, ép toàn bộ sang chuỗi
    # st.dataframe(df_fixed)

    # df["Ngày"] = pd.to_datetime(df["Ngày"]).astype("string")
    # #--- 1. Hiển thị dữ liệu thô---
    st.header("Xem dữ liệu thô")
    st.dataframe(df.head())
    st.subheader("📊 Thông tin cột dữ liệu")   
    st.write(f"👉 Dữ liệu có {df.shape[0]} dòng, {df.shape[1]} cột")
    dtypes_df = df.dtypes.astype(str).reset_index()
    dtypes_df.columns = ["Cột", "Kiểu dữ liệu"]
    st.dataframe(dtypes_df)
    #Kiểm tra giá trị thiếu
    st.subheader("***Kiểm tra giá trị thiếu***")
    describe_null = df.isnull().sum().reset_index()
    describe_null.columns = ["Cột", "Số giá trị thiếu"]
    st.dataframe(describe_null)
    # Thống kê mô tả cơ bản
    st.subheader("📈 Mô tả dữ liệu (describe)")
    st.write(df.describe(include="all").transpose())
    st.subheader("📊 Thống kê mô tả")
    desc = df[["Số lượng", "Giá", "Giảm giá (%)", "Doanh thu"]].describe()
    st.dataframe(desc)
    mean_value = df["Doanh thu"].mean()
    median_value = df["Doanh thu"].median()
    st.write(f"👉 Mean Doanh thu: {mean_value} VND")
    st.write(f"👉 Median Doanh thu: {median_value} VND")

    #--- 3. Histogram Doanh thu ---
    st.header("Histogram Doanh thu- Phân phối Doanh thu")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["Doanh thu"], bins=30, kde=True, ax=ax)
    ax.axvline(mean_value, color="red", linestyle="--", label=f"Mean = {mean_value:,.0f}")
    ax.axvline(median_value, color="green", linestyle="--", label=f"Median = {median_value:,.0f}")
    ax.set_title("Phân phối Doanh thu với Mean và Median")
    ax.legend()
    st.pyplot(fig)
    #--- 4. Boxplot Doanh thu ---
    st.header("Boxplot Doanh thu - Phát hiện ngoại lệ")
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.boxplot(x=df["Doanh thu"], ax=ax2)
    ax2.set_title("Boxplot Doanh thu (Phát hiện ngoại lệ)")
    st.pyplot(fig2)
    #--- 5. Scatter plot Số lượng vs Doanh thu ---
    st.header("Scatter plot Số lượng vs Doanh thu")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x="Số lượng", y="Doanh thu", data=df, ax=ax3)
    ax3.set_title("Số lượng vs Doanh thu")
    st.pyplot(fig3)
    #--- 6. Bar chart - Doanh thu theo khu vực ---
    st.header("Bar chart - Doanh thu theo khu vực")
    revenue_by_region = df.groupby("Khu vực")["Doanh thu"].sum().reset_index()
    fig4, ax4 = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Khu vực", y="Doanh thu", data=revenue_by_region, ax=ax4)
    ax4.set_title("Tổng Doanh thu theo Khu vực")
    st.pyplot(fig4)
    #--- 7. Line chart - Doanh thu theo thời gian ---
    st.header("Line chart - Doanh thu theo thời gian")
    df["Ngày"] = pd.to_datetime(df["Ngày"], errors='coerce')
    revenue_over_time = df.groupby("Ngày")["Doanh thu"].sum().reset_index()
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    sns.lineplot(x="Ngày", y="Doanh thu", data=revenue_over_time, ax=ax5)
    ax5.set_title("Tổng Doanh thu theo Thời gian")
    st.pyplot(fig5)
    #---7. Tương tác nhập dữ liệu---
    st.header("Tính thử Doanh thu dự kiến")
    st.subheader("Nhập dữ liệu bán hàng mới:")
    col1, col2, col3 = st.columns(3)
    with col1:
        so_luong = st.number_input("Số lượng sản phẩm", min_value=1, max_value=100, value=10)
    with col2:
        gia = st.number_input("Giá sản phẩm (VND)", min_value=1000, max_value=50000000, value=1000000, step=1000)
    with col3:
        giam_gia = st.slider("Giảm giá (%)", 0, 30, 10)
    if st.button("Tính doanh thu dự kiến"):
        doanh_thu_mau = so_luong * gia * (1 - giam_gia/100)
        st.write(f"💰 **Doanh thu dự kiến**: {doanh_thu_mau:,.0f} VND")
        fig5, ax5 = plt.subplots(figsize=(8,4))
        sns.histplot(df["Doanh thu"], bins=30, kde=True, ax=ax5, color="lightblue")
        ax5.axvline(mean_value, color="red", linestyle="--", label=f"Mean = {mean_value:,.0f}")
        ax5.axvline(median_value, color="green", linestyle="--", label=f"Median = {median_value:,.0f}")
        ax5.axvline(doanh_thu_mau, color="blue", linestyle="-", label=f"Mẫu nhập = {doanh_thu_mau:,.0f}")
        ax5.set_title("Vị trí doanh thu mẫu so với Mean & Median")
        ax5.legend()
        st.pyplot(fig5)
     # --- 8. Phân tích theo thời gian ---
    st.header("8️⃣ Phân tích theo thời gian")
    df["Năm"] = df["Ngày"].dt.year
    df["Tháng"] = df["Ngày"].dt.month

    option = st.radio("Chọn cách phân tích:", ["Theo năm", "Theo tháng"])

    if option == "Theo năm":
        year_rev = df.groupby("Năm")["Doanh thu"].sum().reset_index()
        fig6, ax6 = plt.subplots(figsize=(6,4))
        sns.barplot(x="Năm", y="Doanh thu", data=year_rev, ax=ax6)
        ax6.set_title("Tổng Doanh thu theo Năm")
        st.pyplot(fig6)
    else:
        selected_year = st.selectbox("Chọn năm:", df["Năm"].unique())
        month_rev = df[df["Năm"]==selected_year].groupby("Tháng")["Doanh thu"].sum().reset_index()
        fig7, ax7 = plt.subplots(figsize=(8,4))
        sns.lineplot(x="Tháng", y="Doanh thu", data=month_rev, marker="o", ax=ax7)
        ax7.set_title(f"Doanh thu theo Tháng - Năm {selected_year}")
        st.pyplot(fig7)
    # --- 9. Dự đoán Doanh thu bằng sklearn ---
    st.header("9️⃣ Dự đoán Doanh thu (Machine Learning)")

    # Chọn các cột để dự báo
    X = df[["Số lượng", "Giá", "Giảm giá (%)", "Danh mục", "Nhà cung cấp", "Khu vực", "Khách hàng", "Phương thức thanh toán"]]
    y = df["Doanh thu"]

    # Xử lý biến categorical
    categorical_cols = ["Danh mục", "Nhà cung cấp", "Khu vực", "Khách hàng", "Phương thức thanh toán"]
    numeric_cols = ["Số lượng", "Giá", "Giảm giá (%)"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    # Pipeline mô hình
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Huấn luyện
    model.fit(X_train, y_train)

    # Đánh giá
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.write("**Hiệu năng mô hình:**")
    st.write(f"R²: {r2:.3f}")
    st.write(f"MAE: {mae:,.0f} VND")
    st.write(f"RMSE: {rmse:,.0f} VND")

    # --- Form nhập dữ liệu để dự đoán ---
    st.subheader("🔮 Thử nhập dữ liệu để dự đoán Doanh thu")

    col1, col2 = st.columns(2)
    with col1:
        so_luong_in = st.number_input("Số lượng", min_value=1, max_value=100, value=5)
        gia_in = st.number_input("Giá sản phẩm (VND)", min_value=1000, max_value=50000000, value=2000000)
        giam_gia_in = st.slider("Giảm giá (%)", 0, 30, 5)
    with col2:
        danh_muc_in = st.selectbox("Danh mục", df["Danh mục"].unique())
        nha_cc_in = st.selectbox("Nhà cung cấp", df["Nhà cung cấp"].unique())
        khu_vuc_in = st.selectbox("Khu vực", df["Khu vực"].unique())
        khach_hang_in = st.selectbox("Khách hàng", df["Khách hàng"].unique())
        pttt_in = st.selectbox("Phương thức thanh toán", df["Phương thức thanh toán"].unique())

    # Tạo dataframe input
    input_data = pd.DataFrame({
        "Số lượng": [so_luong_in],
        "Giá": [gia_in],
        "Giảm giá (%)": [giam_gia_in],
        "Danh mục": [danh_muc_in],
        "Nhà cung cấp": [nha_cc_in],
        "Khu vực": [khu_vuc_in],
        "Khách hàng": [khach_hang_in],
        "Phương thức thanh toán": [pttt_in]
    })

    # Dự đoán
    y_pred_input = model.predict(input_data)[0]
    st.success(f"📈 Doanh thu dự đoán: {y_pred_input:,.0f} VND")
    # # --- Biểu đồ đánh giá mô hình ---
    # st.subheader("📊 Biểu đồ đánh giá mô hình")

    # # Scatter: Thực tế vs Dự đoán
    # fig8, ax8 = plt.subplots(figsize=(6,6))
    # sns.scatterplot(x=y_test, y=y_pred, alpha=0.5, ax=ax8)
    # ax8.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    # ax8.set_xlabel("Doanh thu thực tế")
    # ax8.set_ylabel("Doanh thu dự đoán")
    # ax8.set_title("Scatter: Thực tế vs Dự đoán")
    # st.pyplot(fig8)

    # # Residual plot
    # residuals = y_test - y_pred
    # fig9, ax9 = plt.subplots(figsize=(6,4))
    # sns.scatterplot(x=y_pred, y=residuals, alpha=0.5, ax=ax9)
    # ax9.axhline(0, color="red", linestyle="--")
    # ax9.set_xlabel("Doanh thu dự đoán")
    # ax9.set_ylabel("Phần dư (Residuals)")
    # ax9.set_title("Residual plot")
    # st.pyplot(fig9)

    # # Histogram phần dư
    # fig10, ax10 = plt.subplots(figsize=(6,4))
    # sns.histplot(residuals, bins=30, kde=True, ax=ax10)
    # ax10.set_title("Phân phối phần dư (Residuals)")
    # st.pyplot(fig10)

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích dữ liệu.")
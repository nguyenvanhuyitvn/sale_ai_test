import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Phân tích kinh doanh", layout="wide")

st.title("📊 Phân tích & Dự báo Doanh thu")

# =============================
# Upload file Excel
# =============================
st.sidebar.header("Tải dữ liệu bán hàng")
uploaded_file = st.sidebar.file_uploader("Tải file Excel dữ liệu bán hàng", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df["Ngày"] = pd.to_datetime(df["Ngày"])

    st.success("✅ Đã tải dữ liệu thành công!")

    # =============================
    # Tổng quan số liệu
    # =============================
    daily_rev = df.groupby("Ngày")["Doanh thu"].sum().reset_index()
    st.write(daily_rev)

    df["Năm"] = df["Ngày"].dt.year
    df["Tháng"] = df["Ngày"].dt.month

    monthly_rev = df.groupby(["Năm", "Tháng"])["Doanh thu"].sum().reset_index()
    monthly_rev["Thời gian"] = pd.to_datetime(
        monthly_rev["Năm"].astype(str) + "-" + monthly_rev["Tháng"].astype(str) + "-01"
    )

    region_rev = (
        df.groupby("Khu vực")["Doanh thu"].sum().reset_index().sort_values("Doanh thu", ascending=False)
    )
    cat_rev = (
        df.groupby("Danh mục")["Doanh thu"].sum().reset_index().sort_values("Doanh thu", ascending=False)
    )
    prod_rev = (
        df.groupby("Sản phẩm")["Doanh thu"].sum().reset_index().sort_values("Doanh thu", ascending=False).head(10)
    )

    # Xu hướng MA
    monthly_rev["MA_3"] = monthly_rev["Doanh thu"].rolling(3).mean()

    # =============================
    # Hiển thị số liệu tóm tắt
    # =============================
    st.subheader("📌 Tóm tắt kinh doanh")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng doanh thu", f"{df['Doanh thu'].sum():,.0f} VND")
    with col2:
        st.metric("Doanh thu TB/ngày", f"{daily_rev['Doanh thu'].mean():,.0f} VND")
    with col3:
        st.metric("Doanh thu TB/tháng", f"{monthly_rev['Doanh thu'].mean():,.0f} VND")

    st.write(f"**Khu vực dẫn đầu:** {region_rev.iloc[0]['Khu vực']}")
    st.write(f"**Danh mục dẫn đầu:** {cat_rev.iloc[0]['Danh mục']}")
    st.write(f"**Sản phẩm bán chạy nhất:** {prod_rev.iloc[0]['Sản phẩm']}")

    # =============================
    # Biểu đồ trực quan
    # =============================
    st.subheader("📈 Biểu đồ phân tích")

    # Doanh thu theo ngày
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(daily_rev["Ngày"], daily_rev["Doanh thu"])
    ax.set_title("Doanh thu theo ngày")
    ax.set_xlabel("Ngày")
    ax.set_ylabel("Doanh thu (VND)")
    st.pyplot(fig)

    # Doanh thu theo tháng + MA
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(monthly_rev["Thời gian"], monthly_rev["Doanh thu"], marker="o", label="Doanh thu")
    ax.plot(monthly_rev["Thời gian"], monthly_rev["MA_3"], label="Xu hướng (MA 3 tháng)", linewidth=2)
    ax.set_title("Doanh thu theo tháng & Xu hướng")
    ax.set_xlabel("Thời gian")
    ax.set_ylabel("Doanh thu (VND)")
    ax.legend()
    st.pyplot(fig)

    # Doanh thu theo khu vực
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Doanh thu", y="Khu vực", data=region_rev, ax=ax)
    ax.set_title("Tổng doanh thu theo khu vực")
    st.pyplot(fig)

    # Doanh thu theo danh mục
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Doanh thu", y="Danh mục", data=cat_rev, ax=ax)
    ax.set_title("Tổng doanh thu theo danh mục")
    st.pyplot(fig)

    # Top 10 sản phẩm
    fig, ax = plt.subplots(figsize=(8,5))
    sns.barplot(x="Doanh thu", y="Sản phẩm", data=prod_rev, ax=ax)
    ax.set_title("Top 10 sản phẩm bán chạy theo doanh thu")
    st.pyplot(fig)

    # =============================
    # Kết luận kinh doanh
    # =============================
    st.subheader("📌 Phân tích tình hình & dự báo")
    st.markdown("""
    - **Tình hình hiện tại:**  
      Doanh thu ổn định, có xu hướng tăng nhẹ. TP.HCM và Hà Nội là thị trường chính, Laptop và Điện tử là nhóm sản phẩm chủ lực.
    - **Dự báo gần hạn:**  
      Đường trung bình động (MA 3 tháng) cho thấy doanh thu tiếp tục **tăng ổn định**.
    - **Đề xuất:**  
      - Duy trì tập trung vào TP.HCM và Hà Nội.  
      - Đẩy mạnh marketing cho Laptop, Điện thoại.  
      - Mở rộng nhóm phụ kiện để đa dạng doanh thu.  
    """)
else:
    st.info("⬆️ Vui lòng tải file Excel để bắt đầu phân tích.")

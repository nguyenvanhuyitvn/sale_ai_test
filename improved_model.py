import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# =============================
# 1. Cấu hình ứng dụng và CSS tùy chỉnh
# =============================
st.set_page_config(page_title="Dự đoán Doanh thu Nâng cao", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Ứng dụng Dự đoán Doanh thu Thông minh")
st.markdown("""
<style>
    .stMetric {
        border: 1px solid #2e3b4e;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# =============================
# 2. Các hàm xử lý (ĐÃ BỎ CACHING)
# =============================
def load_data(uploaded_file):
    """Đọc và tiền xử lý dữ liệu từ file Excel."""
    try:
        df = pd.read_excel(uploaded_file)
        # Kiểm tra các cột cần thiết
        required_cols = ["Ngày", "Giá", "Số lượng", "Doanh thu", "Danh mục", "Sản phẩm"]
        if not all(col in df.columns for col in required_cols):
            st.error(f"Lỗi: File Excel phải chứa các cột bắt buộc: {', '.join(required_cols)}")
            return None
            
        df["Ngày"] = pd.to_datetime(df["Ngày"])
        return df
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi đọc file: {e}")
        return None

def feature_engineering(df):
    """Tạo các đặc trưng mới từ dữ liệu thô."""
    df_processed = df.copy()
    df_processed["Tháng"] = df_processed["Ngày"].dt.month
    df_processed["Quý"] = df_processed["Ngày"].dt.quarter
    df_processed["Ngày_trong_tuần"] = df_processed["Ngày"].dt.dayofweek # 0=Thứ 2
    df_processed["Là_cuối_tuần"] = df_processed["Ngày_trong_tuần"].isin([5, 6]).astype(int)
    return df_processed

def train_model(df, model_choice):
    """Chuẩn bị dữ liệu, huấn luyện và trả về mô hình đã train."""
    numeric_features = ["Giá", "Số lượng", "Tháng", "Quý", "Ngày_trong_tuần", "Là_cuối_tuần"]
    
    # Tự động xác định các cột phân loại
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Kiểm tra xem có cột nào không hợp lệ trong list không
    for col in ["Ngày"]: 
        if col in categorical_features:
            categorical_features.remove(col)

    X = df[numeric_features + categorical_features]
    y = df["Doanh thu"]

    # Tạo pipeline tiền xử lý
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder='passthrough'
    )

    # Lựa chọn mô hình
    if model_choice == "Linear Regression":
        regressor = LinearRegression()
    elif model_choice == "Random Forest":
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test, numeric_features, categorical_features

# =============================
# 3. Giao diện chính
# =============================
uploaded_file = st.file_uploader("⬆️ Tải file Excel dữ liệu bán hàng của bạn", type=["xlsx", "xls"])

if uploaded_file is None:
    st.info("💡 Vui lòng tải lên một file Excel để bắt đầu. File cần có các cột như 'Ngày', 'Giá', 'Số lượng', 'Doanh thu', 'Sản phẩm', 'Khu vực'...")
else:
    df_raw = load_data(uploaded_file)
    
    if df_raw is not None:
        st.success("✅ Đã tải và xử lý dữ liệu thành công!")
        df = feature_engineering(df_raw)
        # ---------------------------
        # Sidebar cho các tùy chọn
        # ---------------------------
        with st.sidebar:
            st.header("⚙️ Tùy chọn Mô hình")
            model_choice = st.selectbox(
                "Chọn mô hình AI để dự đoán:",
                ("Linear Regression", "Random Forest")
            )
            st.info(f"Bạn đã chọn: **{model_choice}**. Mô hình sẽ được huấn luyện trên 80% dữ liệu.")

        # ---------------------------
        # Huấn luyện mô hình
        # ---------------------------
        model, X_test, y_test, numeric_features, categorical_features = train_model(df, model_choice)
        y_pred = model.predict(X_test)

        # ---------------------------
        # Tabbed Layout
        # ---------------------------
        tab1, tab2, tab3 = st.tabs(["📊 Tổng quan & Đánh giá Mô hình", "🔮 Dự đoán Đơn hàng Mới", "💡 Phân tích Sâu"])

        with tab1:
            st.header("Tổng quan Kinh doanh")
            col1, col2, col3 = st.columns(3)
            col1.metric("Tổng Doanh thu", f"{df['Doanh thu'].sum():,.0f} VND")
            col2.metric("Tổng Số lượng Bán", f"{df['Số lượng'].sum():,.0f}")
            col3.metric("Đơn hàng Trung bình", f"{df['Doanh thu'].mean():,.0f} VND")

            st.header(f"Đánh giá Hiệu suất Mô hình: {model_choice}")
            col1, col2, col3 = st.columns(3)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            col1.metric("R-squared (R²)", f"{r2:.2f}")
            col2.metric("Mean Absolute Error (MAE)", f"{mae:,.0f} VND")
            col3.metric("Root Mean Squared Error (RMSE)", f"{rmse:,.0f} VND")
            
            st.markdown(f"""
            - **R-squared ($R^2$)**: Mô hình của bạn giải thích được **{r2:.1%}** sự biến thiên của doanh thu. Càng gần 1 càng tốt.
            - **MAE**: Trung bình, dự đoán của mô hình sai lệch khoảng **{mae:,.0f} VND** so với thực tế.
            - **RMSE**: Cho thấy độ lớn của sai số, nhấn mạnh các lỗi dự đoán lớn.
            """)
            
            # Biểu đồ so sánh
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, ax=ax)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Đường tham chiếu (Thực tế = Dự đoán)")
            ax.set_xlabel("Doanh thu Thực tế", fontsize=12)
            ax.set_ylabel("Doanh thu Dự đoán", fontsize=12)
            ax.set_title(f"So sánh Doanh thu Thực tế và Dự đoán ({model_choice})", fontsize=14)
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

        with tab2:
            # st.header("📝 Dự báo Doanh thu cho Kịch bản Mới")
            # with st.form("predict_form"):
            #     st.subheader("Nhập thông tin đơn hàng")
                
            #     col1, col2 = st.columns(2)
            #     with col1:
            #         gia = st.number_input("Giá sản phẩm (VND)", min_value=1000, value=500000, step=1000)
            #         soluong = st.number_input("Số lượng", min_value=1, value=10)
            #         ngay_du_kien = st.date_input("Ngày giao dịch dự kiến")
                
            #     with col2:
            #         # Dynamically create selectboxes for categorical features
            #         input_data_cat = {}
            #         for feature in categorical_features:
            #             unique_vals = df[feature].unique()
            #             input_data_cat[feature] = st.selectbox(f"Chọn {feature}", unique_vals)
                
            #     submitted = st.form_submit_button("Dự đoán Doanh thu")

            # if submitted:
            #     # Chuẩn bị dataframe đầu vào
            #     input_data = {
            #         "Giá": gia,
            #         "Số lượng": soluong,
            #         "Tháng": ngay_du_kien.month,
            #         "Quý": (ngay_du_kien.month - 1) // 3 + 1,
            #         "Ngày_trong_tuần": ngay_du_kien.weekday(),
            #         "Là_cuối_tuần": 1 if ngay_du_kien.weekday() >= 5 else 0
            #     }
            #     input_data.update(input_data_cat)
            #     print(input_data_cat)
            #     print(input_data)
            #     input_df = pd.DataFrame([input_data])
                
            #     # Sắp xếp lại cột cho đúng thứ tự
            #     input_df = input_df[numeric_features + categorical_features]

            #     prediction = model.predict(input_df)[0]
            #     st.success(f"**Dự đoán Doanh thu:** `{prediction:,.0f} VND`")
            #     st.balloons()
            with tab2:
                st.header("📝 Dự báo Doanh thu cho Kịch bản Mới")
                with st.form("predict_form"):
                    st.subheader("Nhập thông tin đơn hàng")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        gia = st.number_input("Giá sản phẩm (VND)", min_value=1000, value=500000, step=1000)
                        soluong = st.number_input("Số lượng", min_value=1, value=10)
                        ngay_du_kien = st.date_input("Ngày giao dịch dự kiến")
                    
                    with col2:
                        # Dynamically create selectboxes for categorical features
                        input_data_cat = {}
                        for feature in categorical_features:
                            unique_vals = df[feature].unique()
                            input_data_cat[feature] = st.selectbox(f"Chọn {feature}", unique_vals)
                    
                    submitted = st.form_submit_button("Dự đoán Doanh thu")

                if submitted:
                    # Chuẩn bị dataframe đầu vào
                    input_data = {
                        "Giá": gia,
                        "Số lượng": soluong,
                        "Tháng": ngay_du_kien.month,
                        "Quý": (ngay_du_kien.month - 1) // 3 + 1,
                        "Ngày_trong_tuần": ngay_du_kien.weekday(),
                        "Là_cuối_tuần": 1 if ngay_du_kien.weekday() >= 5 else 0
                    }
                    input_data.update(input_data_cat)
                    input_df = pd.DataFrame([input_data])
                    
                    # Sắp xếp lại cột cho đúng thứ tự
                    input_df = input_df[numeric_features + categorical_features]

                    prediction = model.predict(input_df)[0]
                    st.success(f"**Dự đoán Doanh thu:** `{prediction:,.0f} VND`")
                    st.balloons()
                    
                    # --- PHẦN BỔ SUNG BIỂU ĐỒ BẮT ĐẦU ---

                    st.markdown("---")
                    st.subheader("📊 Trực quan hóa kết quả dự đoán")

                    col1, col2 = st.columns(2)

                    with col1:
                        # 1. Biểu đồ Cột so sánh
                        st.markdown("##### So sánh với Doanh thu Trung bình")
                        avg_revenue = df['Doanh thu'].mean()
                        
                        fig, ax = plt.subplots()
                        sns.barplot(
                            x=['Doanh thu Trung bình', 'Doanh thu Dự đoán'],
                            y=[avg_revenue, prediction],
                            palette=['skyblue', 'lightgreen'],
                            ax=ax
                        )
                        ax.set_ylabel("Doanh thu (VND)")
                        ax.ticklabel_format(style='plain', axis='y')
                        # Thêm giá trị trên mỗi cột
                        for p in ax.patches:
                            ax.annotate(f"{p.get_height():,.0f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                                        ha='center', va='center', xytext=(0, 9), textcoords='offset points')
                        st.pyplot(fig)

                    with col2:
                        # 2. Biểu đồ Đường "What-If" theo số lượng
                        st.markdown("##### Phân tích nếu thay đổi Số lượng")
                        
                        # Tạo một dải số lượng để dự đoán, ví dụ từ 1 đến 50
                        quantity_range = np.arange(max(1, soluong - 20), soluong + 20, 1)
                        what_if_predictions = []
                        
                        # Lặp qua từng giá trị số lượng để dự đoán
                        for qty in quantity_range:
                            temp_df = input_df.copy()
                            temp_df['Số lượng'] = qty
                            pred = model.predict(temp_df)[0]
                            what_if_predictions.append(pred)
                            
                        fig, ax = plt.subplots()
                        ax.plot(quantity_range, what_if_predictions, marker='o', linestyle='--', label='Xu hướng Doanh thu')
                        # Đánh dấu điểm dự đoán hiện tại
                        ax.plot(soluong, prediction, marker='*', markersize=15, color='red', label=f'Dự đoán hiện tại ({soluong} sp)')
                        ax.set_xlabel("Số lượng sản phẩm")
                        ax.set_ylabel("Doanh thu Dự đoán (VND)")
                        ax.legend()
                        ax.grid(True)
                        ax.ticklabel_format(style='plain', axis='y')
                        st.pyplot(fig)

                    # --- PHẦN BỔ SUNG BIỂU ĐỒ KẾT THÚC ---
        
        with tab3:
            st.header("Phân tích Dữ liệu Khám phá (EDA)")
            
            st.subheader("Doanh thu theo Tháng")
            monthly_revenue = df.groupby("Tháng")["Doanh thu"].sum()
            fig, ax = plt.subplots()
            monthly_revenue.plot(kind='bar', ax=ax, color=sns.color_palette("viridis", 12))
            ax.set_ylabel("Tổng Doanh thu (VND)")
            ax.set_xlabel("Tháng")
            ax.ticklabel_format(style='plain', axis='y')
            st.pyplot(fig)

            st.subheader("Top 10 Sản phẩm có Doanh thu cao nhất")
            top_products = df.groupby("Sản phẩm")["Doanh thu"].sum().nlargest(10)
            fig, ax = plt.subplots()
            top_products.sort_values().plot(kind='barh', ax=ax, color=sns.color_palette("magma", 10))
            ax.set_xlabel("Tổng Doanh thu (VND)")
            ax.set_ylabel("Sản phẩm")
            st.pyplot(fig)
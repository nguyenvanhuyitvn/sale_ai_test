import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

warnings.filterwarnings("ignore")

# =============================
# 1. Cấu hình ứng dụng
# =============================
st.set_page_config(page_title="Dự báo Doanh thu Dòng sản phẩm", layout="wide")
st.title("🚀 Dự báo Doanh thu Tương lai cho Dòng sản phẩm")
st.write("Ứng dụng này sử dụng mô hình SARIMA để phân tích dữ liệu quá khứ và dự báo doanh thu trong các kỳ tiếp theo.")

# =============================
# 2. Các hàm xử lý
# =============================
@st.cache_data
def load_and_preprocess_data(uploaded_file, category_col, date_col, revenue_col):
    """Đọc, kiểm tra và tiền xử lý dữ liệu."""
    try:
        df = pd.read_excel(uploaded_file)
        if not all(col in df.columns for col in [category_col, date_col, revenue_col]):
            st.error(f"Lỗi: File phải chứa các cột '{category_col}', '{date_col}', và '{revenue_col}'.")
            return None, None
        
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col)
        return df, df[category_col].unique()
    except Exception as e:
        st.error(f"Lỗi khi đọc file: {e}")
        return None, None

@st.cache_data
def aggregate_data(df, selected_category, time_freq, category_col, date_col, revenue_col):
    """Lọc và tổng hợp dữ liệu theo chu kỳ thời gian."""
    df_filtered = df[df[category_col] == selected_category]
    
    # Đặt cột ngày làm chỉ mục và tổng hợp doanh thu
    time_series = df_filtered.set_index(date_col)[revenue_col].resample(time_freq).sum()
    
    # Điền giá trị 0 cho các kỳ không có doanh thu
    # --- ĐÃ SỬA LỖI Ở DÒNG DƯỚI ĐÂY ---
    time_series = time_series.asfreq(freq=time_freq, fill_value=0)
    return time_series

# =============================
# 3. Giao diện chính
# =============================
uploaded_file = st.file_uploader("⬆️ Tải file Excel dữ liệu bán hàng của bạn", type=["xlsx", "xls"])

if uploaded_file:
    # --- Sidebar: Nơi người dùng nhập các tùy chọn ---
    st.sidebar.header("⚙️ Cài đặt Phân tích")
    
    # Cho phép người dùng chọn tên các cột quan trọng
    st.sidebar.info("Vui lòng chọn tên các cột tương ứng trong file của bạn.")
    date_col = st.sidebar.text_input("Cột Ngày tháng", "Ngày")
    revenue_col = st.sidebar.text_input("Cột Doanh thu", "Doanh thu")
    category_col = st.sidebar.text_input("Cột Dòng sản phẩm/Danh mục", "Danh mục")
    
    df, categories = load_and_preprocess_data(uploaded_file, category_col, date_col, revenue_col)

    if df is not None:
        st.sidebar.markdown("---")
        st.sidebar.header("🎯 Tùy chọn Dự báo")
        
        selected_category = st.sidebar.selectbox("1. Chọn Dòng sản phẩm để dự báo:", categories)
        
        time_freq = st.sidebar.radio(
            "2. Chọn chu kỳ phân tích:",
            ('Tháng (M)', 'Quý (Q)', 'Tuần (W)'),
            key='time_freq_radio'
        )
        # Lấy mã chu kỳ: 'M', 'Q', 'W'
        freq_code = time_freq.split('(')[1][0]
        
        forecast_steps = st.sidebar.slider("3. Chọn số kỳ muốn dự báo trong tương lai:", 1, 36, 12)
        
        if st.sidebar.button("Bắt đầu Dự báo!", use_container_width=True):
            st.success(f"Đang tiến hành phân tích và dự báo cho: **{selected_category}**...")

            # 1. Tổng hợp dữ liệu
            time_series = aggregate_data(df, selected_category, freq_code, category_col, date_col, revenue_col)
            
            if len(time_series) < 24: # Yêu cầu tối thiểu dữ liệu để mô hình chạy tốt
                 st.warning("Dữ liệu có ít hơn 24 kỳ. Kết quả dự báo có thể không chính xác cao.")
            
            # 2. Chia tab hiển thị
            tab1, tab2 = st.tabs(["📊 Phân tích Chuỗi thời gian", "🔮 Kết quả Dự báo"])

            # --- Tab 1: Phân tích các thành phần của chuỗi thời gian ---
            with tab1:
                st.header("Phân tích các thành phần của Doanh thu")
                st.write("Biểu đồ này tách doanh thu trong quá khứ thành 3 thành phần chính:")
                st.markdown("- **Trend (Xu hướng):** Hướng đi chung của doanh thu (tăng, giảm hay đi ngang).")
                st.markdown("- **Seasonal (Tính mùa vụ):** Các quy luật lặp đi lặp lại theo chu kỳ (ví dụ: tháng nào cũng cao điểm).")
                st.markdown("- **Residual (Phần dư):** Những biến động nhiễu, ngẫu nhiên không giải thích được.")

                try:
                    decomposition = seasonal_decompose(time_series, model='additive', period=12 if freq_code=='M' else 4)
                    fig_decompose = decomposition.plot()
                    fig_decompose.set_size_inches(12, 8)
                    st.pyplot(fig_decompose)
                except Exception as e:
                    st.error(f"Không thể phân rã chuỗi thời gian. Có thể do dữ liệu quá ngắn hoặc không có tính mùa vụ rõ ràng. Lỗi: {e}")

            # --- Tab 2: Dự báo tương lai ---
            with tab2:
                st.header(f"Dự báo Doanh thu cho {forecast_steps} kỳ tiếp theo")

                # 3. Huấn luyện mô hình SARIMA
                # (p,d,q): các tham số cho phần không mùa vụ
                # (P,D,Q,m): các tham số cho phần mùa vụ (m=12 cho tháng, 4 cho quý)
                m = 12 if freq_code == 'M' else (4 if freq_code == 'Q' else 52)
                model = SARIMAX(time_series, order=(1, 1, 1), seasonal_order=(1, 1, 1, m))
                try:
                    results = model.fit(disp=False)
                    
                    # 4. Lấy kết quả dự báo
                    forecast = results.get_forecast(steps=forecast_steps)
                    pred_mean = forecast.predicted_mean
                    conf_int = forecast.conf_int()

                    # 5. Vẽ biểu đồ dự báo
                    fig_forecast, ax = plt.subplots(figsize=(14, 7))
                    
                    # Vẽ dữ liệu lịch sử
                    ax.plot(time_series.index, time_series, label='Dữ liệu Lịch sử')
                    
                    # Vẽ đường dự báo
                    ax.plot(pred_mean.index, pred_mean, color='red', label='Đường Dự báo')
                    
                    # Vẽ vùng tin cậy (vùng không chắc chắn của dự báo)
                    ax.fill_between(conf_int.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.5, label='Vùng tin cậy 95%')

                    ax.set_title(f"Dự báo Doanh thu cho '{selected_category}'", fontsize=16)
                    ax.set_xlabel("Thời gian")
                    ax.set_ylabel("Doanh thu")
                    ax.legend()
                    ax.grid(True)
                    st.pyplot(fig_forecast)

                    # 6. Hiển thị bảng dữ liệu dự báo
                    st.subheader("Bảng dữ liệu dự báo chi tiết")
                    forecast_df = pd.DataFrame({
                        'Doanh thu Dự báo': pred_mean,
                        'Dự báo Thấp (Lower CI)': conf_int.iloc[:, 0],
                        'Dự báo Cao (Upper CI)': conf_int.iloc[:, 1]
                    })
                    st.dataframe(forecast_df.style.format("{:,.0f} VND"))

                except Exception as e:
                    st.error(f"Gặp lỗi khi huấn luyện hoặc dự báo: {e}")
                    st.info("Vui lòng thử lại với chuỗi dữ liệu dài hơn hoặc kiểm tra dữ liệu đầu vào.")
else:
    st.info("⬆️ Vui lòng tải lên một file Excel để bắt đầu.")
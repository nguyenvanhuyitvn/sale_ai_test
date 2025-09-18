# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error

# # Giả lập dữ liệu phi tuyến: y = x^2 + noise
# np.random.seed(42)
# X = np.random.rand(100, 1) * 10
# y = X.squeeze()**2 + np.random.randn(100) * 5

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 1. Linear Regression
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)
# y_pred_lin = lin_reg.predict(X_test)

# # 2. RandomForest Regressor
# rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_reg.fit(X_train, y_train)
# y_pred_rf = rf_reg.predict(X_test)

# # So sánh lỗi
# print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lin))
# print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))

import numpy as np
from sklearn.model_selection import train_test_split

# X = np.arange(10).reshape((5, 2))  # 5 mẫu, mỗi mẫu có 2 đặc trưng
# y = [0, 1, 2, 3, 4]                # nhãn tương ứng

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# print("X:", X)
# print("y:", y)
# print("X_train:", X_train)
# print("y_train:", y_train)
# print("X_test:", X_test)
# print("y_test:", y_test)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Online Python - IDE, Editor, Compiler, Interpreter
TV = [[10,20,30,40,50]]
Sales = [[1500, 3500, 2500, 5000, 1000]]
df = pd.DataFrame({"TV": TV, "Sales" : Sales})

X = df["TV"].values
y = df["Sales"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
# Hệ số
beta1 = model.coef_[0]      # slope
beta0 = model.intercept_   # intercept

# Dự đoán
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
y_pred_all   = model.predict(X)

# Metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse  = mean_squared_error(y_test, y_pred_test)
train_r2  = r2_score(y_train, y_pred_train)
test_r2   = r2_score(y_test, y_pred_test)

print("Hệ số hồi quy (β1 - slope):", beta1)
print("Giao điểm (β0 - intercept):", beta0)
print()
print("Train MSE:", train_mse)
print("Test MSE:", test_mse)
print("Train R²:", train_r2)
print("Test R²:", test_r2)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

df = pd.read_csv("FuelConsumption.csv")

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)

print('cofficients', regr.coef_)
print('intercept', regr.intercept_)

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print("R2-score: %.2f" % r2_score(test_y, test_y_))

# دریافت مقدار حجم موتور از کاربر
engine_size = float(input("لطفاً حجم موتور را وارد کنید: "))

# تبدیل مقدار ورودی به آرایه مناسب برای مدل
engine_size_array = np.array([[engine_size]])

# استفاده از مدل برای پیش‌بینی مقدار CO2
predicted_co2 = regr.predict(engine_size_array)

# نمایش نتیجه
print(f"میزان پیش‌بینی‌شده انتشار CO2: {predicted_co2[0][0]:.2f}")

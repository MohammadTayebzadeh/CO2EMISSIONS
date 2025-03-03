import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import r2_score

# بارگذاری داده‌ها از فایل CSV
df = pd.read_csv("FuelConsumption.csv")

# انتخاب ستون‌های مرتبط با مدل
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]

# تقسیم داده‌ها به دو مجموعه آموزشی و تست (80% آموزش و 20% تست)
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# ایجاد مدل رگرسیون خطی
regr = linear_model.LinearRegression()

# انتخاب داده‌های مربوط به حجم موتور (برای ویژگی‌ها) و انتشار CO2 (برای هدف)
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

# آموزش مدل روی داده‌های آموزشی
regr.fit(train_x, train_y)

# نمایش ضرایب مدل (شیب خط رگرسیون) و عرض از مبدأ
print('cofficients', regr.coef_)
print('intercept', regr.intercept_)

# انتخاب داده‌های تست برای ارزیابی مدل
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

# پیش‌بینی انتشار CO2 برای داده‌های تست
test_y_ = regr.predict(test_x)

# محاسبه و نمایش R2-score برای ارزیابی دقت مدل
print("R2-score: %.2f" % r2_score(test_y, test_y_))

# دریافت مقدار حجم موتور از کاربر
engine_size = float(input("لطفاً حجم موتور را وارد کنید: "))

# تبدیل مقدار ورودی به آرایه مناسب برای مدل
engine_size_array = np.array([[engine_size]])

# استفاده از مدل برای پیش‌بینی مقدار CO2
predicted_co2 = regr.predict(engine_size_array)

# نمایش نتیجه پیش‌بینی
print(f"میزان پیش‌بینی‌شده انتشار CO2: {predicted_co2[0][0]:.2f}")

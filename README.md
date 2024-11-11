# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 11.11.24

### AIM:
To Create a project on Time series analysis on weather forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of weather 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```
# Step 1: Load your dataset (replace 'your_data.csv' with the actual filename)
# Assuming the dataset is already loaded into df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data provided
df= pd.read_csv('supermarketsales.csv')
# Convert 'Date' to datetime and set it as the index for time series analysis
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Step 2: Check for stationarity (ADF Test, Differencing, and Plotting)
# ADF test
def adf_test(series):
    result = adfuller(series)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("Series is stationary")
    else:
        print("Series is not stationary")

print("ADF Test on 'Total':")
adf_test(df['Total'])

# Plot ACF and PACF
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(acf(df['Total'], nlags=35))
plt.title('ACF Plot')
plt.subplot(122)
plt.plot(pacf(df['Total'], nlags=35))
plt.title('PACF Plot')
plt.show()

# Differencing if needed - first reset the index to avoid duplicate index errors
df.reset_index(inplace=True)

# Calculate the differenced series and drop NaNs, then reassign 'Date' as index
df['Total_diff'] = df['Total'].diff().dropna()
df.set_index('Date', inplace=True)

print("ADF Test on Differenced 'Total':")
adf_test(df['Total_diff'].dropna())

# Continue with further steps based on your algorithm
# Example parameters for ARIMA
p, d, q = 1, 1, 1  # example values

# Step 4: Fit the ARIMA model
model = ARIMA(df['Total'], order=(p, d, q))
model_fit = model.fit()
print(model_fit.summary())

# Step 5: Make time series predictions
forecast = model_fit.get_forecast(steps=30)
forecast_index = pd.date_range(df.index[-1], periods=30, freq='D')
forecast_df = forecast.summary_frame(alpha=0.05)
forecast_df.index = forecast_index

# Plot the forecast
plt.figure(figsize=(12, 6))
plt.plot(df['Total'], label='Original Data')
plt.plot(forecast_df['mean'], label='Forecast')
plt.fill_between(forecast_df.index, forecast_df['mean_ci_lower'], forecast_df['mean_ci_upper'], color='pink', alpha=0.3)
plt.legend()
plt.show()

```

### OUTPUT:

![image](https://github.com/user-attachments/assets/ca29cb99-877e-4388-9e7e-7182d59eaa36)

![image](https://github.com/user-attachments/assets/fefc61c1-d966-455e-b0f6-e24924709f29)

![image](https://github.com/user-attachments/assets/27ff6043-6198-4722-bb6c-6e685725b6d5)




### RESULT:
Thus the program run successfully based on the ARIMA model using python.

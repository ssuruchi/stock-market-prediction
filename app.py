import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as fyf
import pandas_datareader as data
from pandas_datareader import data as pdr
from keras.models import load_model
import streamlit as st


start = '2007-01-01'
end = '2022-12-01'

st.title("Stock Market Prediction")

fyf.pdr_override()

user_input = st.text_input("Enter Stock Ticker", "SBIN.NS")
df = pdr.get_data_yahoo(user_input,start, end)

# Describing Data
st.subheader("Data from 2007 to 2022")
st.write(df.describe())

# Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with MA100')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with MA200')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with MA100 and MA200')
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)

# Splitting Data into Training and Testing Data
training_data = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
testing_data = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

print(training_data.shape)
print(testing_data.shape)

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))

training_data_arr = scalar.fit_transform(training_data)

# Load the model
model = load_model('keras_model.h5')

past_100_days = training_data.tail(100)
final_df = pd.concat([past_100_days, pd.DataFrame(testing_data)], ignore_index =True)
input_data = scalar.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scalar.scale_

scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Load
st.subheader("Predicted vs Original")
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label  ='Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

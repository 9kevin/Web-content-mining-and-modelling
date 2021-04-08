import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader as web
import tensorflow as tf
import datetime as dt
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

model = tf.keras.models.load_model("stock_model.sav")
scaler = MinMaxScaler(feature_range=(0,1))
companyName = 'FB'
startDate = dt.datetime(2010,1,1)
endDate = dt.datetime(2020,1,1)
data = web.DataReader(companyName, 'yahoo', startDate, endDate)
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

@st.cache
def load_data():
  
  prediction_days = 60
  test_start = dt.datetime(2020,1,2)
  test_end = dt.datetime.now()
  test_data = web.DataReader(companyName, 'yahoo', test_start, test_end)
  columns = test_data.columns

  # Preprocess data
  actual_prices = test_data['Close'].values
  full_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)
  model_inputs = full_dataset[len(full_dataset) - len(test_data) - prediction_days:].values
  model_inputs = model_inputs.reshape(-1, 1)
  model_inputs = scaler.transform(model_inputs)

  # Preparing data for predicting the next day
  real_data = np.array([model_inputs[len(model_inputs)+1-prediction_days : len(model_inputs+1), 0]])
  real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
  return actual_prices, test_data, columns, real_data

def predict_stock(real_data):
  predict = [[264.9187]]
  try:
    predict = model.predict(real_data)
  except ValueError:
    pass  # do nothing
  prediction_scaled = scaler.inverse_transform(predict)
  return prediction_scaled


actual_prices, data, columns, real_data = load_data()
st.title("Predict Facebook Stock Prices")
html_temp = """
<div style="background-color:tomato;padding:10px">
<h3 style="color:white;">We help you predict the Facebook closing stock price for the next day</h3>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

check_box = st.sidebar.checkbox(label="Display the dataset")
st.sidebar.title("Options")
if check_box:
  st.write(data)

feature_selection = st.sidebar.multiselect(label="Select the features you want to visualize", options=columns)
if feature_selection:
  df_feature = data[feature_selection]
  plotly_figure = px.line(data_frame=data, x=df_feature.index, y=feature_selection, title=('Facebook Stock Prices'))
  st.plotly_chart(plotly_figure)

r = ""
if st.button("Predict the stock market closing value for tomorrow"):
    result = predict_stock(real_data)
    r = result[0][0]
st.success('The closing stock price for facebook is predicted to be : {}'.format(r))

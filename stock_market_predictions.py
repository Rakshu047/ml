import pandas as pd
import numpy as np
import sklearn.metrics as mts
from pandas_datareader import data as pdr
from datetime import datetime,date 
import streamlit as st
import yfinance as yf

yf.pdr_override()

start_date=st.sidebar.date_input('Start Date : ',value=date(2000,1,1))
today=date.today()
end_date=st.sidebar.date_input('Start Date : ',value=today)

user_input=st.text_input('Enter the Stock Ticker: ','AAPL')


try:
    df=pdr.get_data_yahoo(user_input, start=start_date, end=end_date)
    st.write("Stock Data")
    st.write(df)
except Exception as e:
    st.write("Error:",e)


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

#df=pdr.get_data_yahoo(user_input, start_date, end_date)

#descrribing stock data
#st.subheader("Stock Data")
#st.write(df.describe())

try:
    df=pdr.get_data_yahoo(user_input, start=start_date, end=end_date)
    st.write("Stock Data")
    st.write(df)
except Exception as e:
    st.write("Error:",e)

df.isnull().sum()
df=df.drop([Date],axis=1)

from sklearn.impute import SimpleImputer
x=df.drop(['Close'],axis=1)
imp=SimpleImputer(missing_values=0,strategy='mean')
imp=imp.fit(x)
x=imp.transform(x)

y=df['Close']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.05, random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

pred=reg.predict(x_test)

from sklearn.metrics import accuracy_score
print("accuracy score : ",accuracy_score(pred,y_test))
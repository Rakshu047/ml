import pandas as pd
from pandas_datareader import data as pdr
from datetime import date 
import streamlit as st
import yfinance as yf

yf.pdr_override()

st.title("Stock Market Prediction")

start_date=st.sidebar.date_input('Start Date : ',value=date(2000,1,1))
today=date.today()
end_date=st.sidebar.date_input('Start Date : ',value=today)

user_input=st.text_input('Enter the Stock Ticker: ','AAPL')

df=pdr.get_data_yahoo(user_input, start=start_date, end=end_date)

try:
    st.write("Stock Data")
    st.write(df.head(100))
except Exception as e:
    st.write("Error:",e)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12,6))
ma100 = df['Close'].rolling(100).mean()
ma200 = df['Close'].rolling(200).mean()
plt.plot(df['Close'])
plt.plot(ma100, 'g')
plt.plot(ma200, 'r')
plt.xlabel('Date')
plt.ylabel('Close Value')
plt.title('Actual Close Values Over Time')
st.pyplot(fig)

df["Tomorrow"] = df['Close'].shift(-1)
df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, min_samples_split = 50, random_state = 1)

train = df.iloc[:-100]
test = df.iloc[-100:]

predictors = ["Open", "High", "Low", "Close", "Volume"]

model.fit(train[predictors],train["Target"])

horizons = [2,5,60,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = df.rolling(horizon).mean()
    
    ratio_column = f"Close_ratio_{horizon}"
    df[ratio_column] = df["Close"]/rolling_averages["Close"]
    
    trend_column = f"Trend_{horizon}"
    df[trend_column] = df.shift(1).rolling(horizon).sum()["Target"]
    
    new_predictors += [ratio_column, trend_column]

df = df.dropna()

def predict(train, test, predictors, model):
    model.fit(train[predictors],train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >= .75] = 1
    preds[preds < .75] = 0
    preds = pd.Series(preds, index=test.index, name = "Predictions")
    combined = pd.concat([test["Target"],preds], axis = 1)
    return combined

def backtest(data, model, predictors, start=2500, step=250):
    all_predictions = []
    
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i]
        test = data.iloc[i:(i+step)]
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.DataFrame(pd.concat(all_predictions))

predictions = backtest(df, model, new_predictors)

if predictions["Predictions"][:,-1] == 1:
    st.write("The Stock price will Increase")
else:
    st.write("The Stock price will decrease")

# This code is a Python script that uses various libraries and modules to perform stock market
# prediction. Here is a breakdown of what the code does:
import pandas as pd
from pandas_datareader import data as pdr
from datetime import date 
import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# `yf.pdr_override()` is a function from the `yfinance` library that overrides the default behavior of
# the `pandas_datareader` library. It allows us to use `yfinance` as the data source for fetching
# stock market data instead of the default source. This is necessary because the default source for
# `pandas_datareader` has been deprecated.
yf.pdr_override()

st.title("Stock Market Prediction")
start_date=st.sidebar.date_input('Start Date : ',value=date(2000,1,1))
today=date.today()
end_date=st.sidebar.date_input('End Date : ',value=today)
user_input=st.text_input('Enter the Stock Ticker: ')

# This code block is responsible for fetching stock market data from Yahoo Finance based on the user's
# input.
if st.sidebar.button("Fetch Data"):
    df=pdr.get_data_yahoo(user_input, start=start_date, end=end_date)

    try:
        st.write("Stock Data")
        st.write(df.head(100))
    except Exception as e:
        st.write("Error:",e)

# This code block is creating a figure object using `plt.figure()` and specifying the size of the
# figure using the `figsize` parameter. Then, it plots the closing values of the stock (`df['Close']`)
# using `plt.plot()`. It sets the x-axis label to 'Date' using `plt.xlabel()`, the y-axis label to
# 'Close Value' using `plt.ylabel()`, and the title of the plot to 'Closing Value V/S Date' using
# `plt.title()`. Finally, it displays the plot using `st.pyplot(fig)`, which is a function from the
# `streamlit` library that allows the plot to be displayed in the Streamlit app.
    fig = plt.figure(figsize=(16,6))
    plt.plot(df['Close'])
    plt.xlabel('Date')
    plt.ylabel('Close Value')
    plt.title('Clsoing Value V/S Date ')
    st.pyplot(fig)

    ma100 = df['Close'].rolling(100).mean()
    fig = plt.figure(figsize=(16,6))
    plt.plot(df['Close'])
    plt.plot(ma100, 'g')
    plt.xlabel('Date')
    plt.ylabel('Close Value')
    plt.title('Clsoing Value V/S Date with 100 MA')
    st.pyplot(fig)

    ma200 = df['Close'].rolling(200).mean()
    fig = plt.figure(figsize=(16,6))
    plt.plot(df['Close'])
    plt.plot(ma100, 'g')
    plt.plot(ma200, 'r')
    plt.xlabel('Date')
    plt.ylabel('Close Value')
    plt.title('Clsoing Value V/S Date with 100 MA & 200 MA')
    st.pyplot(fig)

    
# This code block is preparing the data for training a machine learning model to predict stock market
# trends.
 # The code `df["Tomorrow"] = df['Close'].shift(-1)` creates a new column in the DataFrame `df` called
 # "Tomorrow". It shifts the values of the "Close" column one step forward (-1) in order to represent
 # the closing price of the next day.
    df["Tomorrow"] = df['Close'].shift(-1)
    df['Target'] = (df['Tomorrow'] > df['Close']).astype(int)
    model = RandomForestClassifier(n_estimators = 200, min_samples_split = 50, random_state = 1)
    train = df.iloc[:-100]
    test = df.iloc[-100:]
    
    predictors = ["Open", "High", "Low", "Close", "Volume"]

    model.fit(train[predictors],train["Target"])
    
# The code block you mentioned is calculating rolling averages and creating new columns in the
# DataFrame `df` based on different time horizons.
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
        """
        The function `predict` takes in training and testing data, a list of predictors, and a model, fits
        the training data to the model, makes predictions on the testing data, and returns a combined
        dataframe of the actual target values and the predicted values.
        
        :param train: The training dataset, which contains the features used to train the model and the
        corresponding target variable
        :param test: The test parameter is a DataFrame containing the test data. It should have the same
        columns as the train DataFrame, except for the "Target" column which is the column we want to
        predict
        :param predictors: The predictors parameter is a list of column names from the train and test
        datasets that will be used as input features for the model. These features will be used to predict
        the target variable
        :param model: The "model" parameter refers to the machine learning model that will be used for
        prediction. It could be any model that has a `fit` method for training and a `predict_proba` method
        for making predictions. Examples of such models include logistic regression, random forest,
        gradient boosting, etc
        :return: a combined DataFrame that contains the "Target" column from the test DataFrame and the
        "Predictions" column, which is the result of applying the model to the test data.
        """
        model.fit(train[predictors],train["Target"])
        preds = model.predict_proba(test[predictors])[:,1]
        preds[preds >= .75] = 1
        preds[preds < .75] = 0
        preds = pd.Series(preds, index=test.index, name = "Predictions")
        combined = pd.concat([test["Target"],preds], axis = 1)
        return combined

    def backtest(data, model, predictors, start=1000, step= 250):
        """
        The `backtest` function performs a backtest on a given model using a specified set of predictors and
        returns the predictions for each test set.
        
        :param data: The data parameter is a pandas DataFrame that contains the historical data for
        backtesting. It should have columns for the features (predictors) and the target variable (the
        variable we want to predict)
        :param model: The `model` parameter refers to the machine learning model that will be used for
        prediction. It could be any model that supports the `fit` and `predict` methods, such as linear
        regression, random forest, or support vector machines
        :param predictors: The "predictors" parameter is a list of features or variables that will be used
        as inputs to the model for making predictions. These features are used to train the model and then
        used to make predictions on the test data
        :param start: The `start` parameter is the index position where the backtesting should start. It
        specifies the starting point of the training data. By default, it is set to 1000, which means the
        backtesting will start from the 1000th index position of the `data` DataFrame, defaults to 1000
        (optional)
        :param step: The `step` parameter in the `backtest` function determines the number of data points to
        include in each test set. It specifies the size of the sliding window used for testing the model's
        performance. In this case, the `step` parameter is set to 250, which means that the, defaults to 250
        (optional)
        :return: a pandas DataFrame that contains all the predictions made during the backtesting process.
        """
        all_predictions = []
        for i in range(start, data.shape[0], step):
            train = data.iloc[0:i]
            test = data.iloc[i:(i+step)]
            predictions = predict(train, test, predictors, model)
            all_predictions.append(predictions)
        return pd.DataFrame(pd.concat(all_predictions))

# The code block you mentioned is using the `backtest` function to make predictions on the test data
# using the trained machine learning model.
    predictions = backtest(df, model, new_predictors)
    test_df = predictions.tail(1)
    if (test_df["Predictions"] == 1.0).bool():
        st.sidebar.write("The Stock price will Increase")
    else:
        st.sidebar.write("The Stock price will decrease")
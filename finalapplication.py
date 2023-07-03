import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import date
from sklearn.model_selection import train_test_split
from pandas.tseries.offsets import BDay
import plotly.io as pio
from xgboost import XGBRegressor
from PIL import Image
from model_loader import load_model_past
from model_loader import load_model_future

# Streamlit web app
def main():
    st.title("EUR/USD Currency Value Prediction")
    
    
    tab1, tab2, tab3, tab4 = st.tabs(["About", "EDA", "Past Prediction", "Future Prediction"])

    with tab1:
        st.header("About Project")

        image = Image.open('eurodollar.jpg')
        st.image(image)

        # Display bullet points
        st.markdown("- This project aims to provide users with assistance in making informed decisions regarding the buying or selling of forex assets by utilizing prediction techniques.")
        st.markdown("- The project was developed as part of a Data Science Internship.")
        st.markdown("- The chosen model for this project is XGBoost, a powerful machine learning algorithm.")
        st.markdown("- It is important to note that relying solely on the model's predictions is not advised. There are limitations that should be considered. Financial markets are complex and influenced by various factors, making it challenging to accurately capture them in a predictive model. Additionally, unforeseen events and sudden market fluctuations can affect the model's performance.")

    with tab2:
        st.header("Exploratory Data Analysis (EDA)")
        st.write("Choose the visualization you want to preview first:")
        
        # Select the visualization option
        visualization_option = st.selectbox(
            'Select Visualization',
            ('Line Graph of Closing Price', 'Daily Returns', 'Candlestick Chart', 'Moving Averages (50-day and 200-day)', 'Volatility Visualization (Average True Range)')
        )
        
        # Set the symbol and start date
        symbol = 'EURUSD=X'
        start_date = pd.Timestamp('2013-01-01')
        end_date = pd.Timestamp.today()  # Updated to current date

        # Download the data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date)

        # Reset the index
        data = data.reset_index()
        
        # Set the template to 'plotly_dark' for a dark theme
        pio.templates.default = "plotly_dark"

        if visualization_option == 'Line Graph of Closing Price':
            # Plot the closing price over time
            fig = px.line(data, x='Date', y='Close', title='EUR/USD Closing Price')
            fig.update_layout(xaxis_rangeslider_visible=True)
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Closing Price')
            st.plotly_chart(fig)
            
            st.divider()
            st.write
        elif visualization_option == 'Daily Returns':
            # Calculate and plot the daily returns
            data['Return'] = data['Close'].pct_change()
            fig = px.line(data, x='Date', y='Return', title='EUR/USD Daily Returns')
            fig.update_layout(xaxis_rangeslider_visible=True)
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='Daily Return')
            st.plotly_chart(fig)

        elif visualization_option == 'Candlestick Chart':
            # Convert 'Date' column to datetime format
            data['Date'] = pd.to_datetime(data['Date'])

            # Candlestick Chart
            fig = go.Figure(data=go.Candlestick(x=data['Date'].astype(str),
                                                open=data['Open'],
                                                high=data['High'],
                                                low=data['Low'],
                                                close=data['Close']))

            fig.update_layout(title='EUR/USD Candlestick Chart',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            xaxis_rangeslider_visible=True)

            st.plotly_chart(fig)

            
        elif visualization_option == 'Moving Averages (50-day and 200-day)':
            # Calculate the moving averages
            data['MA_50'] = data['Close'].rolling(window=50).mean()
            data['MA_200'] = data['Close'].rolling(window=200).mean()

            # Plot the closing price and moving averages
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['MA_50'], name='50-day Moving Average'))
            fig.add_trace(go.Scatter(x=data['Date'], y=data['MA_200'], name='200-day Moving Average'))

            fig.update_layout(title='EUR/USD Moving Averages',
                              xaxis_title='Date',
                              yaxis_title='Price')
            fig.update_layout(xaxis_rangeslider_visible=True)
            st.plotly_chart(fig)
            
        elif visualization_option == 'Volatility Visualization (Average True Range)':
            # Calculate the Average True Range (ATR)
            data['ATR'] = data['High'] - data['Low']

            # Plot the ATR
            fig = px.line(data, x='Date', y='ATR', title='EUR/USD Average True Range')
            fig.update_layout(xaxis_rangeslider_visible=True)
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='ATR')
            st.plotly_chart(fig)
        
    

    with tab3:
        st.header("Past Prediction")

        # Load the pre-trained XGBoost model
        #model = XGBRegressor()
        #model.load_model("past_xgboost_model.json.json")

        # Load the pre-trained XGBoost model
        model = load_model_past()


        # Set the currency
        symbol = "EURUSD=X"

        # Allow user to select training data date range
        # start_date_input = st.date_input("Start Date", value=pd.Timestamp('2013-01-01'))
        # end_date_input = st.date_input("End Date", value=pd.Timestamp('2022-12-31'))

        st.write("The training data is from 01/01/2013 untill 31/12/2022")
        start_date_input = value=pd.Timestamp('2013-01-01')
        end_date_input = value=pd.Timestamp('2022-12-31')

        # Download the currency data
        forex_data = yf.download(symbol, start=start_date_input, end=end_date_input)

        # Prepare the data for training
        forex_close = forex_data['Close']
        forex_data = forex_close.to_numpy()

        # Normalize the data between 0 and 1
        min_value = np.min(forex_data)
        max_value = np.max(forex_data)
        forex_data = (forex_data - min_value) / (max_value - min_value)

        # Define the sequence length
        seq_length = 30

        step = 1  # Step size for creating sequences

        X = []
        y = []

        for i in range(0, len(forex_data) - seq_length, step):
            X.append(forex_data[i:i+seq_length])
            y.append(forex_data[i+seq_length])

        X = np.array(X)
        y = np.array(y)

        # Use all the data for training
        X_train, y_train = X, y

        # Create and train the XGBoost model
        model = XGBRegressor()
        model.fit(X_train, y_train)

        # Allow user to select future prediction date range
        prediction_start_date_input = st.date_input("Prediction Start Date")
        prediction_end_date_input = st.date_input("Prediction End Date")

        # Add Predict button
        if st.button("Predict"):
            # Download actual data for the selected prediction date range
            actual_data = yf.download(symbol, start=prediction_start_date_input, end=prediction_end_date_input)['Close']

            # Prepare the data for prediction
            last_sequence = np.expand_dims(forex_data[-seq_length:], axis=0)  # Last sequence in the available data

            # Make predictions for the selected prediction date range
            future_predictions = []

            for i in range(len(actual_data)):
                prediction = model.predict(last_sequence)
                future_predictions.append(prediction[0])
                last_sequence = np.append(last_sequence, np.expand_dims(prediction, axis=0), axis=1)
                last_sequence = np.delete(last_sequence, 0, axis=1)

            # Denormalize the predictions
            future_predictions = np.array(future_predictions) * (max_value - min_value) + min_value

            # Calculate the differences between actual and predicted values
            differences = np.abs(actual_data - future_predictions)

            # Create a DataFrame with the prediction results
            prediction_results = pd.DataFrame({
                "Actual": actual_data,
                "Predicted": future_predictions,
                "Difference": differences
            }).round(4)

            # Plot the actual and predicted values
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual_data.index, y=actual_data, mode='lines', name='Actual'))
            fig.add_trace(go.Scatter(x=actual_data.index, y=future_predictions, mode='lines', name='Predicted'))
            fig.update_layout(title='EUR/USD Actual vs Predicted', xaxis_title='Date', yaxis_title='Closing Price')
            st.plotly_chart(fig)

            # Display the prediction results
            st.dataframe(prediction_results.style.set_properties(**{'width': 'auto', 'max-width': '500px'}))
            
            
    with tab4:
        st.header("Future Prediction")
        
        # Load the pre-trained XGBoost model
        #with open('xgboost_model.h5', 'rb') as file:
        #    model = pickle.load(file)

        # Load the pre-trained XGBoost model
        #model = xgb.Booster()
        #model.load_model('future_xgboost_model.json.json')

        # Load the pre-trained XGBoost model
        model = load_model_future()

        # Define the prediction durations and their corresponding options
        prediction_durations = {
            'week': 7,
            'month': 30,
            'year': 365
        }
        
        # Select the prediction duration
        duration = st.radio('Select Prediction Duration', list(prediction_durations.keys()))
        num_days = prediction_durations[duration]
        
        # Download the currency data
        start_date = pd.Timestamp('2013-01-01')
        end_date = pd.Timestamp.today()  # Updated to current date
        forex_data = yf.download('EURUSD=X', start=start_date, end=end_date)
        
        # Prepare the data for training
        forex_close = forex_data['Close']
        
        # Define the appropriate sequence length
        if duration == 'week':
            seq_length = 7
        elif duration == 'month':
            seq_length = 30
        else:
            seq_length = 365
        
        step = 1  # Step size for creating sequences
        
        X = []
        y = []
        
        for i in range(0, len(forex_close) - seq_length, step):
            X.append(forex_close[i:i+seq_length])
            y.append(forex_close[i+seq_length])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        
        # Create and train the XGBoost model
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1, random_state=123)
        model.fit(X_train, y_train)
        
        # Make predictions for the next 'num_days' days
        future_predictions = []
        X_pred = forex_close[-seq_length:].values
        
        for _ in range(num_days):
            prediction = model.predict(X_pred.reshape(1, seq_length))
            future_predictions.append(prediction[0])
            X_pred = np.append(X_pred[1:], prediction)
        
        # Create the timeline for the future predictions (using business days)
        last_date = forex_close.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.DateOffset(days=1), periods=num_days)

        # Plot the predicted values
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forex_close.index, y=forex_close, mode='lines', name='Actual', visible='legendonly'))
        fig.add_trace(go.Scatter(x=future_dates, y=future_predictions, mode='lines', name='Predicted'))

        fig.update_layout(title='EUR/USD Value Prediction for the Next {} Days'.format(num_days),
                        xaxis_title='Date', yaxis_title='Currency Value')

        st.plotly_chart(fig)


# Run the web app
if __name__ == '__main__':
    main()

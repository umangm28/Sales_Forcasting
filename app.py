from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load the data and model (the loading process you have already implemented)
df = pd.read_csv("data/grouped_data_timeperiod.csv")
df['TimePeriod'] = pd.to_datetime(df['TimePeriod'])
monthly_sales = df.groupby(['TimePeriod'])['Total_Sales_Sum'].sum().reset_index()

# Define your model and scaler
look_back = 12
scaler = MinMaxScaler(feature_range=(0, 1))

# Function to create dataset for LSTM
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        X.append(a)
        Y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(Y)

# Preparing the data
train_size = int(len(monthly_sales) * 0.8)
train, test = monthly_sales.iloc[:train_size], monthly_sales.iloc[train_size:]

train_scaled = scaler.fit_transform(train['Total_Sales_Sum'].values.reshape(-1, 1))
test_scaled = scaler.transform(test['Total_Sales_Sum'].values.reshape(-1, 1))

X_train, y_train = create_dataset(train_scaled, look_back)
X_test, y_test = create_dataset(test_scaled, look_back)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build and load your LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=20)

# Function to forecast future values


def forecast_future(steps, start_input):
    future_predictions = np.array([])
    future_input = start_input

    for _ in range(steps):
        future_input = future_input.reshape((1, look_back, 1))
        pred = model.predict(future_input)
        future_predictions = np.concatenate((future_predictions, pred.flatten()))
        future_input = np.append(future_input[:, 1:, :], pred)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))


last_date = monthly_sales['TimePeriod'].iloc[-1]
future_steps = 6
future_predictions = forecast_future(future_steps, test_scaled[-look_back:])
future_dates = pd.date_range(last_date, periods=future_steps, freq='MS')[0:]
print(future_predictions,future_dates)

# last_date = monthly_sales['TimePeriod'].iloc[-1]
future_steps_2 = 12
future_predictions_2 = forecast_future(future_steps_2, test_scaled[-look_back:])
future_dates_2 = pd.date_range(future_dates[-1], periods=future_steps_2 + 1, freq='MS')[1:]
print(future_predictions_2,future_dates_2)

# Define routes for the API
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/aprsep', methods=['GET'])
def aprsep():
    
    future_predictions_df = pd.DataFrame({'Date': future_dates, 'Predicted_Total_Sales': future_predictions.flatten()})
    return jsonify(future_predictions_df[(future_predictions_df['Date'] >= '2024-04-01') & (future_predictions_df['Date'] <= '2024-09-01')].to_dict(orient='records'))

@app.route('/yearly', methods=['GET'])
def yearly():
    
    future_predictions_df_2 = pd.DataFrame({'Date': future_dates_2, 'Predicted_Total_Sales': future_predictions_2.flatten()})
    return jsonify(future_predictions_df_2[(future_predictions_df_2['Date'] >= '2024-10-01') & (future_predictions_df_2['Date'] <= '2025-09-01')].to_dict(orient='records'), )

if __name__ == '__main__':
    app.run(debug=True)

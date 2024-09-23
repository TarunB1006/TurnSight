import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def deviation(actual, predicted):
    if actual > 0:
        return min((abs(actual - predicted) / actual), 1)
    elif actual == 0 and predicted > 0:
        return 1
    else:
        return 0

devs = []
# Load your data
turns = ['BC', 'BE', 'DA', 'DE', 'FA', 'FC']
vehicle_classes = ['Car', 'Truck', 'Bus', 'LCV', 'Two-Wheeler', 'Three-Wheeler', 'Bicycle']

for turn in turns:
    train_1 = f'log2/{turn}_60s.csv'
    train_2 = f'log3/{turn}_60s.csv'
    test_1 = f'log4/{turn}_60s.csv'
    test_2 = f'log5/{turn}_60s.csv'

    train_df_1 = pd.read_csv(train_1, dtype=int)
    train_df_2 = pd.read_csv(train_2, dtype=int)
    test_df_1 = pd.read_csv(test_1, dtype=int)
    test_df_2 = pd.read_csv(test_2, dtype=int)

    train_df_2 = train_df_2 + train_df_1.iloc[-1]
    test_df_2 = test_df_2 + test_df_1.iloc[-1]

    train_df = pd.concat([train_df_1, train_df_2], ignore_index=True)
    test_df = pd.concat([test_df_1, test_df_2], ignore_index=True)

    # Get length of training data
    train_len = len(train_df)

    # Adjust the seconds in the test set to start from the end of the training set
    test_df['seconds'] = test_df['seconds'] + 900

    # Adjust the test set counts to start from the end of the training set
    for vehicle in vehicle_classes:
        test_df[vehicle] = test_df[vehicle] + train_df[vehicle].iloc[-1]

    # Convert 'seconds' to datetime format
    train_df['ds'] = pd.to_datetime(train_df['seconds'], unit='s')

    # Prepare the data for Prophet
    for vehicle in vehicle_classes:
        df = train_df[['ds', vehicle]]
        df = df.rename(columns={vehicle: 'y'})

        # Initialize and fit the model
        model = Prophet()
        model.fit(df)

        # Make a future dataframe for predictions
        future = model.make_future_dataframe(periods=train_len, freq='S')
        forecast = model.predict(future)

        # Round the prediction values to integers
        forecast['yhat'] = forecast['yhat'].round().astype(int)

        # Get the actual value from the test set (final value)
        actual_value = test_df[vehicle].iloc[-1]

        # Get the predicted value from the forecast (final value)
        predicted_value = forecast['yhat'].iloc[-1]

        # Print the results
        print(f"Turn: {turn}, Vehicle: {vehicle}")
        print(f"Actual: {actual_value}, Predicted: {predicted_value}")
        print("-" * 40)
        devs.append(deviation(actual_value - train_df[vehicle].iloc[-1], predicted_value - train_df[vehicle].iloc[-1]))

final_pred_dev = final_pred_dev = (sum(devs)/(len(turns)*len(vehicle_classes)))*100
print(f"Final Prediction Deviation: {final_pred_dev:.2f}%")

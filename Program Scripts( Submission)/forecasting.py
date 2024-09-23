import pandas as pd
from collections import Counter
from statsmodels.tsa.arima.model import ARIMA

from pmdarima import auto_arima
import csv


class Forecaster:
    def __init__(self):
        self.vehicle_classes = ['Car', 'Bus', 'Truck', 'Three Wheeler', 'Two Wheeler', 'LCV', 'Bicycle']
   
    def forecast_vehicle_counts(self, cumulative_counts, frame_count_dict, fps):
        predicted_counts = {}
        frame_interval = fps * 15  # 15 seconds interval
        forecast_points = 120  # 30 minutes forecast

        for pattern, frames in frame_count_dict.items():
            filename = f'{pattern}_frame_counts.csv'
            self.save_frame_counts_to_csv(filename, frames)
            df = pd.read_csv(filename)

            if df.empty or df['Frame Number'].isna().all():
                print(f"Warning: The DataFrame for pattern '{pattern}' is empty or contains only NaN values.")
                predicted_counts[pattern] = {cls: 0 for cls in self.vehicle_classes}
                continue

            max_frame_number = df['Frame Number'].max()
            if pd.isna(max_frame_number):
                max_frame_number = 0

            desired_frames = pd.DataFrame({'Frame Number': range(frame_interval, int(max_frame_number) + frame_interval, frame_interval)})
            df_interpolated = pd.merge_asof(desired_frames, df.sort_values('Frame Number'), on='Frame Number')
            df_interpolated.fillna(0, inplace=True)

            forecast_results = {}
            for vehicle_class in df_interpolated.columns[1:]:
                class_data = df_interpolated[vehicle_class]
                try:
                    arima_model = auto_arima(class_data, start_p = 0, start_q = 0, max_p = 5, max_q = 5, seasonal=True, stepwise=False, suppress_warnings=True, error_action="ignore",random=True,n_fits=50,method="lbfgs",maxiter=200)
                    forecast, _ = arima_model.predict(n_periods=forecast_points, return_conf_int=True)
                    forecast_results[vehicle_class] = int(forecast.iloc[-1]) - int(class_data.iloc[-1])
                except Exception as e:
                    print(f"ARIMA fitting failed for {vehicle_class} with error: {e}")
                    forecast_results[vehicle_class] = 0

            predicted_counts[pattern] = forecast_results

        return predicted_counts 
 


    
    def save_frame_counts_to_csv(self, filename, frames):
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['Frame Number'] + self.vehicle_classes
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for frame_number, counts in frames:
                row = {'Frame Number': frame_number}
                row.update(counts)
                writer.writerow(row)

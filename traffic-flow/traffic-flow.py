import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import seaborn as sns
from tqdm import tqdm

# Import Traffic count predictions
traffic_predictions = pd.read_feather('predictions/traffic-counts/pred_traffic_counts_yolov5')
# Import the traffic data to produce median models
clean_birmingham_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_birmingham_report_df_norm')
clean_manchester_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_manchester_report_df_norm')
clean_cambridge_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_cambridge_report_df_norm')
clean_thorpe_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_thorpe_report_df_norm')
clean_epping_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_epping_report_df_norm')
clean_bristol_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_bristol_report_df_norm')
print('\nImported Reports\n')


def mean_model(train_report):
    '''
    Function makes predictions for normalised traffic
    
            Parameters:
                train report: (dataframe) report_df dataframe of traffic values
            
            Return:
                predictions: (dataframe) dataframe of predictions of normalised traffic
    '''
    predictions = train_report.groupby('timestamp')['total_volume_normalised'].mean().to_frame().reset_index()
    return predictions


def median_model(train_report):
    '''
    Does the same as mean_model but uses median instead of mean
    '''
    predictions = train_report.groupby('timestamp')['total_volume_normalised'].median().to_frame().reset_index()
    return predictions


def report_metrics(y_true, y_pred):
    '''
    Output metrics of predictions vs truth
    '''
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R^2: {r2}")
    
    return mse, mae, r2

    
def evaluate_model(predictions, test_report):
    '''
    Evaluates the predictions on test_report
    '''
    true_values = test_report[['timestamp', 'total_volume_normalised']]
    pred_vs_true = pd.merge(true_values, predictions, on="timestamp", how='left')
    pred_vs_true.columns = ['timestamp', 'y_true', 'y_pred']
    # Remove rows wherre there is no prediction, very few of these occur
    pred_vs_true = pred_vs_true[~pred_vs_true.y_pred.isna()]
    # Report the metrics
    mse, mae, r2 = report_metrics(pred_vs_true.y_true,pred_vs_true.y_pred)
    
    return mse, mae, r2


# Global Models
print('\nBirmingham Metrics\n')
train_report = pd.concat([clean_manchester_report_df_norm, clean_cambridge_report_df_norm, clean_thorpe_report_df_norm, clean_epping_report_df_norm, clean_bristol_report_df_norm], ignore_index=True)
test_report = clean_birmingham_report_df_norm

predictions = mean_model(train_report)
results = evaluate_model(predictions, test_report)

# AADT predictions
# you calculate N(t) via predictions[predictions.timestamp==t]
# therefore mean[N(t)] = predictions.mean()
predictions_2021 = predictions[predictions.timestamp.dt.year == 2021]
predictions_2021.total_volume_normalised.mean()

test_report.

print(clean_birmingham_report_df_norm)
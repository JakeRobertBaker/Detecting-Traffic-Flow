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


class Normalised_Flow_Model:
    def __init__(self, train_report):
        self.time_range = pd.date_range("2019-03-19 00:14:00", "2022-04-08 23:59:00", freq="15min")
        # Don't include out of sync time values
        train_report_in_range = train_report[train_report.timestamp.isin(self.time_range)]
        self.predictions = train_report_in_range.groupby('timestamp')['total_volume_normalised'].median().to_frame().reset_index()
        self.true_mu = self.predictions.total_volume_normalised.sum() / len(self.time_range)
        self.empty_times = []
        
        for t in tqdm(self.time_range):
            if len( self.predictions[self.predictions.timestamp==t] ) <1:
                self.empty_times.append(t)
                
        print(f"{len(self.empty_times)} of {len(self.time_range)} time values are empty, {len(self.empty_times)/len(self.time_range) *100} %")
        
    def N(self,t):
        result = self.predictions[self.predictions.timestamp == t]
        if len(result) == 1:
            N_t = result.total_volume_normalised.to_numpy().item()
        else:
            N_t = 0
        return N_t
    
    
    def mean_N(self, date_range):
        N_t_in_date_range = self.predictions[self.predictions.timestamp.isin(date_range)]
        mean_N_t = N_t_in_date_range.total_volume_normalised.sum()/len(date_range)
        return mean_N_t
    
    def mean_N_year(self, year):
        start_time = str(year) + "-01-01 00:14:00"
        end_time = str(year) + "-12-31 23:59:00"
        date_range = pd.date_range(start_time, end_time, freq="15min")
        
        mean_N_t = self.mean_N(date_range)
        return mean_N_t
    
    
# Global Models
print('\nBirmingham Metrics\n')
train_report = pd.concat([clean_manchester_report_df_norm, 
                          clean_cambridge_report_df_norm, 
                          clean_thorpe_report_df_norm, 
                          clean_epping_report_df_norm, 
                          clean_bristol_report_df_norm], ignore_index=True)

test_report = clean_birmingham_report_df_norm

# Predict AADTs for sites in the test report.
AADT_preds = []
AADTs = []
date_range = pd.date_range("2021-01-01 00:14:00", "2021-12-31 23:59:00", freq="15min")
for site in test_report.site_id.unique():
    test_report_site = test_report[(test_report.site_id == site)]
    AADT = test_report_site[test_report_site.timestamp.isin(date_range)].total_volume.sum()/len(date_range)
    #time = time = datetime.datetime(2021,12,8,11,14)
    time = time = datetime.datetime(2021,6,15,11,14)
    X_t = test_report_site[test_report_site.timestamp == time].total_volume.to_numpy()
    AADT_pred = X_t/a.N(time) * a.mean_N_year(2021)
    # print(f"AADT {AADT}, X_t {X_t}, N(t) {a.N(time)}, N_(2021): {a.mean_N_year(2021)}, AADT_pred {AADT_pred}")
    AADTs.append(AADT)
    AADT_preds.append(AADT_pred.item())
    
fig, ax = plt.subplots(figsize=(10,10))
sns.scatterplot(x=AADTs, y=AADT_preds, ax=ax)
        
lims = [
    np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
    np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
]
# now plot both limits against eachother
plt.plot(lims, lims, 'k-', alpha=0.5)
        
plt.show()
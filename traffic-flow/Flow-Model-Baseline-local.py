import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import datetime
import calendar
from collections import defaultdict
from patsy import dmatrices
from patsy import dmatrix
from sklearn.linear_model import LinearRegression
print('imports done')


# Import the traffic data to produce median models
clean_birmingham_report_df_norm = pd.read_feather('high_quality_traffic_reports_two_year/clean_birmingham_report_df_norm')
clean_manchester_report_df_norm = pd.read_feather('high_quality_traffic_reports_two_year/clean_manchester_report_df_norm')
clean_cambridge_report_df_norm = pd.read_feather('high_quality_traffic_reports_two_year/clean_cambridge_report_df_norm')
clean_thorpe_report_df_norm = pd.read_feather('high_quality_traffic_reports_two_year/clean_thorpe_report_df_norm')
clean_epping_report_df_norm = pd.read_feather('high_quality_traffic_reports_two_year/clean_epping_report_df_norm')
clean_bristol_report_df_norm = pd.read_feather('high_quality_traffic_reports_two_year/clean_bristol_report_df_norm')

reports = {"birmingham": clean_birmingham_report_df_norm,
          "manchester" : clean_manchester_report_df_norm,
          "cambridge" : clean_cambridge_report_df_norm,
          "thorpe" : clean_thorpe_report_df_norm,
          "epping" : clean_epping_report_df_norm,
          "bristol" : clean_bristol_report_df_norm }

times = {"birmingham": datetime.datetime(2021,6,15,11,14),
          "manchester" : datetime.datetime(2019,4,19,12,59),
          "cambridge" : datetime.datetime(2021,11,2,11,29),
          "thorpe" : datetime.datetime(2022,3,8,11,14),
          "epping" : datetime.datetime(2022,3,8,11,14),
          "bristol" : datetime.datetime(2020,8,8,11,14) }


# Function that produces the train report
def get_reports(test_loc):
    train_report = pd.concat([v for k,v in reports.items() if k != test_loc])
    return train_report


# Function that creates the time factors
def add_time_data(report):
    report.loc[:,'hour'] = report.timestamp.dt.hour.astype(str)
    report.loc[:,'DOW'] =  report.timestamp.dt.day_of_week.astype(str)
    report.loc[:,'month'] = report.timestamp.dt.month.astype(str)
    return report


def year_str(time):
    '''
    Returns two strings representing the first and last day of time's year
    '''
    start_time = str(time.year) + "-01-01 00:14:00"
    end_time = str(time.year) + "-12-31 23:59:00"
    return start_time, end_time


for area in reports.keys():
    print(area)

    train_report = reports[area]
    start_time, end_time = year_str(times[area])
    train_report = train_report[train_report.timestamp.between(start_time, end_time)]
    
    test_report = pd.DataFrame(pd.date_range(start_time, end_time, freq="15min"))
    test_report.columns = ['timestamp']
    test_report = test_report[test_report.timestamp.between(train_report.timestamp.min(), train_report.timestamp.max())]
    
    print('reports compiled')
    print('train dates')
    print(train_report.timestamp.max())
    print(train_report.timestamp.min())

    print('test dates')
    print(test_report.timestamp.max())
    print(test_report.timestamp.min())

    # Add the time factors to the data
    train_report = add_time_data(train_report)
    test_report = add_time_data(test_report)

    # Filter down onto the model columns
    data_train = train_report[['total_volume_normalised', 'hour', 'DOW', 'month']]
    data_test = test_report[['hour', 'DOW', 'month']]

    # Free up some ram
    del train_report

    # Make Model Matrix
    y_train, X_train = dmatrices("total_volume_normalised ~ month + DOW + hour + DOW*hour", data_train)
    print('train matrix done')

    # Fit model
    model = LinearRegression()
    model.fit(X_train,y_train)
    print('model fitted')

    # Create model matrix for test data
    X_test = dmatrix("month + DOW + hour + DOW*hour", data_test)
    print('test matrix done')

    # Make predictions on test data
    predictions_test = model.predict(X_test)
    print('test predictions done')

    # Put the predictions in the test report
    test_report.loc[:,'total_volume_normalised_predictions'] = predictions_test

    # Export result
    test_report.to_feather(f'linear-{area}-local')
    print('exported')

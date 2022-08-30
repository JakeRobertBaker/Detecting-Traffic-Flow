import tensorflow as tf
import gpflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


print(tf.config.list_physical_devices('GPU'))


# Import count data
clean_birmingham_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean/clean_birmingham_report_df_norm')

# For each timetstamp take the median normalised count value
median_birmingham_report_df_norm = train_report.groupby('timestamp')['total_volume_normalised'].mean().to_frame().reset_index()

print('worked')
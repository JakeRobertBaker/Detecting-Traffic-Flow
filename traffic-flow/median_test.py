import argparse
import pandas as pd


parser = argparse.ArgumentParser(description='Model the traffic flow')

parser.add_argument('--makedata', dest='makedata', action='store_const',
                    const=True, default=False,
                    help='make the median data (default: assume the data is already made)')

args = parser.parse_args()


if args.makedata:
  # Import count data
  clean_birmingham_report_df_norm = pd.read_feather('high_quality_traffic_reports/clean_birmingham_report_df_norm')

  # For each timetstamp take the median normalised count value
  median_birmingham_report_df_norm = clean_birmingham_report_df_norm.groupby('timestamp')['total_volume_normalised'].median().to_frame().reset_index()
  # median_birmingham_report_df_norm['float_time'] = median_birmingham_report_df_norm.timestamp.astype(int)/1E9
  median_birmingham_report_df_norm['float_time'] = (median_birmingham_report_df_norm.timestamp.apply(lambda x: x.value) - 1552954440000000000)/1E9 * 1/900
  
  # Save the dataset
  median_birmingham_report_df_norm.to_feather('high_quality_traffic_reports/median_birmingham_report_df_norm')
  print('Generated the median data')
  
else:
  median_birmingham_report_df_norm = pd.read_feather('high_quality_traffic_reports/median_birmingham_report_df_norm')
  print('Imported the median data')
  
print(median_birmingham_report_df_norm)
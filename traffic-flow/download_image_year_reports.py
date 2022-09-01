import sys
from api import *


# Download information about all UK sites
url_text = "https://webtris.highwaysengland.co.uk/api/v1/sites"
with urllib.request.urlopen(url_text) as url:
    data = json.loads(url.read().decode())
sites = data['sites']

argument = sys.argv[-1]

# Go though the area mentioned in arguments and download reports
if argument in ['birmingham', 'all']:
    print('Querying Birmingham')
    start_date, end_date = '01012021', '31122021'
    max_lat, max_long = 52.50, -1.67
    min_lat, min_long = 52.42, -1.75
    clean_birmingham_report_df_norm, __ = download_clean_pipeline(start_date, end_date, max_lat, max_long, min_lat, min_long, 40)
    clean_birmingham_report_df_norm.reset_index(drop=True).to_feather('image_year_traffic_reports/clean_birmingham_report_df')
    
if argument in ['manchester', 'all']:
    print('Querying Manchester')
    start_date, end_date = '01012019', '31122019'
    max_lat, max_long = 53.51, -2.31
    min_lat, min_long = 53.44, -2.39
    clean_manchester_report_df_norm, __ = download_clean_pipeline(start_date, end_date, max_lat, max_long, min_lat, min_long, 40)
    clean_manchester_report_df_norm.reset_index(drop=True).to_feather('image_year_traffic_reports/clean_manc_report_df')

if argument in ['cambridge', 'all']:
    print('Querying Cambridge')
    start_date, end_date = '01012021', '31122021'
    max_lat, max_long = 52.25, 0.11
    min_lat, min_long = 52.19, 0.02
    clean_cambridge_report_df_norm, __ = download_clean_pipeline(start_date, end_date, max_lat, max_long, min_lat, min_long, 40)
    clean_cambridge_report_df_norm.reset_index(drop=True).to_feather('image_year_traffic_reports/clean_cam_report_df')
    
if argument in ['thorpe', 'all']:
    print('Querying Thorpe')
    start_date, end_date = '01012022', '31122022'
    max_lat, max_long = 51.43, -0.50
    min_lat, min_long = 51.38, -0.57
    clean_thorpe_report_df_norm, __ = download_clean_pipeline(start_date, end_date, max_lat, max_long, min_lat, min_long, 40)
    clean_thorpe_report_df_norm.reset_index(drop=True).to_feather('image_year_traffic_reports/clean_thorpe_report_df')
    
if argument in ['epping', 'all']:
    print('Querying Epping')
    start_date, end_date = '01012022', '31122022'
    max_lat, max_long = 51.72, 0.15
    min_lat, min_long = 51.62, 0.09
    clean_epping_report_df_norm, __ = download_clean_pipeline(start_date, end_date, max_lat, max_long, min_lat, min_long, 40)
    clean_epping_report_df_norm.reset_index(drop=True).to_feather('image_year_traffic_reports/clean_epping_report_df')

if argument in ['bristol', 'all']:
    print('Querying Bristol')
    start_date, end_date = '01012020', '31122020'
    max_lat, max_long = 51.60, -2.52
    min_lat, min_long = 51.52, -2.59
    clean_bristol_report_df_norm, __ = download_clean_pipeline(start_date, end_date, max_lat, max_long, min_lat, min_long, 40)
    clean_bristol_report_df_norm.reset_index(drop=True).to_feather('image_year_traffic_reports/clean_bristol_df')
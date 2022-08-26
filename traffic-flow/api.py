import urllib.request, json
import pandas as pd
import math
import pickle
import collections
from tqdm import tqdm
import geopandas
import datetime



# Download information about all UK sites
url_text = "https://webtris.highwaysengland.co.uk/api/v1/sites"
with urllib.request.urlopen(url_text) as url:
    data = json.loads(url.read().decode())
sites = data['sites']

def get_quality_area(sites,
                     max_lat=math.inf,
                     max_long=math.inf,
                     min_lat=-math.inf,
                     min_long=-math.inf,
                     start_date='01062021',
                     end_date = '15062022',
                     quality_threshold = 90):
    '''
    Returns a dataframe of Traffic Count sites in the specified area and time with sufficent reporting quality

            Parameters:
                    max_lat, max_long, min_lat, min_long (int): Coordinates defining the rectangular area of interest. Default is entire globe.
                    start_date, end_date (str): Strings of the start and end dates of our search ddmmyy
                    quality_threshold (int): Only indclude sites that have at least quality_threshold % of times reporting data

            Returns:
                    quality_area_sites_df (dataframe): Report high quality sites, cols are: Id, Name, Description, Longitude, Latitude, Status
    '''
    # Convert sites query into df and filter onto our area
    sites_df = pd.DataFrame(data = sites)
    area_sites_df = sites_df.loc[(min_long < sites_df.Longitude) & (sites_df.Longitude < max_long)
                                & (min_lat < sites_df.Latitude) & (sites_df.Latitude < max_lat)]
    area_sites_df = area_sites_df.reset_index(drop=True)
    area_ids = list(area_sites_df.Id)
    
    # Next filter onto sites with good quality data:
    quality_responces = []
    for site_id in tqdm(area_ids):
        url_text = f"https://webtris.highwaysengland.co.uk/api/v1/quality/overall?sites={site_id}&start_date={start_date}&end_date={end_date}"
        with urllib.request.urlopen(url_text) as url:
            responce = json.loads(url.read().decode())
        quality_responces.append(responce)
        
    # We only want sites with quality greater than threshold
    good_quality_ids = []
    for responce in quality_responces:
        if responce['data_quality'] >= quality_threshold:
            good_quality_ids.append(responce['sites'])

    quality_area_sites_df = area_sites_df.loc[area_sites_df.Id.isin(good_quality_ids)]
    quality_area_sites_df = quality_area_sites_df.reset_index(drop=True)
    
    return quality_area_sites_df


def daily_report_query_url(site_id, page_num, start_date = '15062021', end_date = '15062022'):
    '''Generates the query url for page page_num of traffic reporting of site site_id'''
    query_url = f"https://webtris.highwaysengland.co.uk/api/v1/reports/Daily?sites={site_id}&start_date={start_date}&end_date={end_date}&page={page_num}&page_size=10000"
    return query_url


def get_site_report(site_id, start_date='15062021', end_date='15062022'):
    '''
    Returns a dataframe of traffic counts on a specified site and date range.

            Parameters:
                    site_id (str): Site's unique id
                    start_date, end_date (str): Strings of the start and end dates of our search ddmmyy

            Returns:
                    report_df (dataframe): Report of traffic counts for that site
                    header (dict): Columns of the dataframe
    '''
    # Download page 1
    report_url = daily_report_query_url(site_id, 1, start_date, end_date)
    with urllib.request.urlopen(report_url) as url:
        report_page = json.loads(url.read().decode())
        
    # Work out how many pages are required    
    header = report_page['Header']
    rows = report_page['Rows']
    row_count = header['row_count']
    total_pages = math.ceil(row_count / 10000)
    # Make a dataframe of the rows so dar
    report_df = pd.DataFrame(data = rows)
    
    for i in range(2, total_pages+1):
        # Get page i of the report
        report_url = daily_report_query_url(site_id, i, start_date, end_date)
        with urllib.request.urlopen(report_url) as url:
            report_page = json.loads(url.read().decode())
        
        rows = report_page['Rows']
        current_page_df = pd.DataFrame(data = rows)
        report_df = pd.concat([report_df, current_page_df], ignore_index=True)

    return report_df, header


def get_reports_from_sites_df(sites_df, start_date, end_date):
    '''
    Returns a dataframe of traffic counts for an entire set of sites

            Parameters:
                    sites_df (dataframe): The sites we want to query, has the same columns as get_quality_area function's output
                    start_date, end_date (str): Strings of the start and end dates of our search ddmmyy

            Returns:
                    report_df (dataframe): Report of traffic counts for the sites
    '''
    # Get the reports on the site
    train_reports =  collections.defaultdict(str)
    # Go through all the site ids and get reports
    for site_id in tqdm(sites_df.Id):
        report, header = get_site_report(site_id, start_date, end_date)
        report['site_id'] = site_id
        train_reports[site_id] = report
        
    # Combine reports into one df
    report_df = pd.concat(list(train_reports.values()), ignore_index=True)
    return report_df


def df_to_gdf(site_df):
    '''Converts a dataframe outputted by get_quality_area into a geodataframe with cordinates'''
    gdf = geopandas.GeoDataFrame(
        site_df, geometry=geopandas.points_from_xy(site_df.Longitude, site_df.Latitude))
    return gdf


def clean_report(report_df):
    '''
    Cleans the traffic count report with a few key steps:
    1. Format the column names and remove redundant columns
    2. Converts the count columns into intergers
    3. Remove rows with blank data
    4. Remove rows that only report one value (zero)
    5. Add a timestamp column to the report
    
            Parameters:
                    report_df (dataframe): The report of traffic count data outputted by get_reports_from_sites_df
            
            Returns:
                    clean_report_df (datafrane): The cleaned report
    '''
    # Step 1.
    clean_col_names = [
        'site_name',
        'report_date',
        'time_period_ending',
        'time_interval',
        '0-520cm',
        '521-660cm',
        '661-1160cm',
        '1160+cm',
        '0-10mph',
        '11-15mph',
        '16-20mph',
        '21-25mph',
        '26-30mph',
        '31-35mph',
        '36-40mph',
        '41-45mph',
        '46-50mph',
        '51-55mph',
        '56-60mph',
        '61-70mph',
        '71-80mph',
        '80+mph',
        'avg_mph',
        'total_volume',
        'site_id']
    report_df.columns = clean_col_names
    clean_cols = [
         'site_name',
         'site_id',
         'report_date',
         'time_period_ending',
         'time_interval',
         '0-520cm',
         '521-660cm',
         '661-1160cm',
         '1160+cm',
         'avg_mph',
         'total_volume']
    clean_report_df = report_df[clean_cols]
    
    # Steps 2., 3., 4.
    interger_cols = [
         '0-520cm',
         '521-660cm',
         '661-1160cm',
         '1160+cm',
         'total_volume']
    def remove_rows(df):
        df = df.loc[df['total_volume'] != '']  # Remove empty rows
        x = df.groupby('site_id')['total_volume'].nunique()
        zero_sites = list(x[x==1].index)  # Remove sites where the volume is always zero
        df = df.loc[~df.site_id.isin(zero_sites)]
        df[interger_cols] = df[interger_cols].astype('int32')
        return df
    clean_report_df = remove_rows(clean_report_df)
    
    # Step 5.
    def get_timestamp(row):
        year, month,day = row['report_date'].split('T')[0].split('-')
        hour, minute, second = row['time_period_ending'].split(':')
        return datetime.datetime(int(year),int(month),int(day), int(hour), int(minute))
    
    clean_report_df['timestamp'] = clean_report_df.apply(get_timestamp,axis=1)
    return clean_report_df


# Function used to normalsise the count data
def normalise(clean_report):
    interger_cols = ['0-520cm', '521-660cm', '661-1160cm', '1160+cm', 'total_volume']
    for name in interger_cols:
        new_name = f"{name}_normalised"
        # for ever row in the report present the row's site id's mean volume
        mean = clean_report.groupby('site_id')[name].transform("mean")
        # normalise
        clean_report.loc[:, new_name] = clean_report[name] / mean
        # filter so we don't have rows with a small mean which causes a pole
    return clean_report[mean>1]


# A pipeline of stages for downloading and normalising reporting.
def download_clean_pipeline(start_date, end_date, max_lat, max_long, min_lat, min_long, quality_threshold = 90):
    # Get the quality data
    sites_df = get_quality_area(sites, max_lat, max_long, min_lat, min_long, start_date, end_date, quality_threshold)
    # Download the report
    report_df = get_reports_from_sites_df(sites_df, start_date, end_date)
    # Clean the report
    clean_report_df = clean_report(report_df)
    # Normalsie the report
    clean_report_df_norm = normalise(clean_report_df)
    return clean_report_df_norm, sites_df
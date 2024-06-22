"""
USGS, DayMET, STATSGO, etc. data grabber
Harrison Myers
4/30/2024

This script takes a USGS gauge ID and automatically generates a dataframe of 
hydrometeorological and catchment attribute data for that gauge. 
"""
import pandas as pd
import numpy as np
import datetime as dt
from pynhd import NLDI
import pydaymet as daymet
import matplotlib.pyplot as plt
from datetime import timedelta, datetime
import time, pickle, os, requests, warnings, traceback

warnings.filterwarnings('ignore')

# Create directories
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
data_dir = dir_path + '\\data'
#%%
class DataGrabber:
    """
    A class to download and compile data from USGS (through an API), GAGESII 
    (stored locally), and DayMET (through an API) to generate a dataset of 
    hydrometeorological and catchment physiographic attributes across all 
    USGS gauges that have the desired parameters.
    """
    def __init__(self, data_dir, param_dict):
        self.data_dir = data_dir
        self.param_dict = param_dict
        self.final_gauge_dict = {}
    
    def get_gage_location(self, staid):
        """
        Returns latitude and longitude of gage
        Params:
            -- staid (str, req) - Station ID 
        Returns:
            -- latitude, longitude (float) - latitude and longitude of USGS gauge
        """
        try:
            response = requests.get(f'https://waterservices.usgs.gov/nwis/site/?format=rdb&sites={staid}')
            lines = response.text.splitlines()
            for line in lines:
                if not line.startswith('#'):
                    parts = line.split('\t')
                    if len(parts) > 4 and parts[1] == staid:
                        latitude = float(parts[4])
                        longitude = float(parts[5])
                        return latitude, longitude
            raise ValueError(f"Location data not found for gage {id}")
        except Exception as e:
            print(f"Failed to retrieve location for gage {id} due to {e}")
        return None, None
    
    def grab_usgs_data(self, id, start, end):
        """
        Form URL for USGS station request and return formatted dataframe of daily 
        data for the given station.
        
        USGS parameter codes
        https://help.waterdata.usgs.gov/codes-and-parameters/parameters
        https://help.waterdata.usgs.gov/parameter_cd?group_cd=PHY
        Streamflow, mean. daily in cubic ft / sec: '00060',
        Gage height, ft: '00065'
        DO, unfiltered, mg/L: '00300',
        Temperature, water, C: '00010'
        
        Args:
        -- id (str) [req]: station ID to get data for
        -- start (datetime) [req]: start datetime
        -- end (datetime) [req]: end datetime

        Returns:
        A dataframe of USGS data with the specified parameters indexed by timestamp
        """
        # Add in functionality to get multiple parameters from USGS website
        if len(self.param_dict) > 1:
            param_str = ','.join(list(self.param_dict.values()))
        else:
            param_str = self.param_dict[0]
            
        # Get location data    
        lat, lon = self.get_gage_location(id)
        
        # put request inside while loop to make sure it is successful
        returnValue = None
        while returnValue is None:
            # try:
            gage = requests.get('https://waterservices.usgs.gov/nwis/dv/'
                                '?format=json'
                                f'&sites={id}'
                                f'&startDT={start.strftime("%Y-%m-%d")}'
                                f'&endDT={(end-dt.timedelta(days=1)).strftime("%Y-%m-%d")}'
                                f'&parameterCd={param_str}'
                                )
            try:
                returnValue = gage.json()
                timeSeries = returnValue['value']['timeSeries']

                # Get parameter data
                param_data = {}
                for series in timeSeries:
                    param_data[series['variable']['variableCode'][0]['value']] = series['values'][0]['value']
                    
            # Optional additional error handling
            # except requests.exceptions.Timeout:
            #     print(f"Timeout occurred for gage {id}. Skipping to the next gage.")
            #     return None
            
            # except requests.exceptions.ConnectionError:
            #     print(f"Connection error occurred for gage {id}. Skipping to next gage.")
            #     return None
            
            except Exception as e:
                print(f"USGS Observational Hydrology Data Request Failed at gage {id} due to {e}... Will retry")
                traceback.print_exc()

        dfs = []
        try:
            for param, values in param_data.items():
                df = pd.DataFrame(values)
                df = pd.DataFrame(data={f'{param}': df['value'].astype(float).values}, index=pd.to_datetime(df['dateTime'], utc=True).dt.date)
                df.index.name = 'date'
                dfs.append(df)
        except: 
            pass
        try:
            station_df = pd.concat(dfs, axis=1)
            
            # rename columns
            col_name_dict = {v: k for k, v in self.param_dict.items()}
            station_df = station_df.rename(columns=col_name_dict)
            
            # Add latitude and longitude to station dataframe
            station_df['lat'] = np.full(len(station_df), lat)
            station_df['lon'] = np.full(len(station_df), lon)
            
            # Sort dataframe index
            station_df = station_df.sort_index()
            
            # Find the first DO index, and start the dataframe there
            first_do_obs = station_df['o'].first_valid_index()
            if first_do_obs is not None:
                station_df = station_df.loc[first_do_obs:]
            else:
                # If all values are NA, return an empty DataFrame
                station_df = pd.DataFrame(columns=station_df.columns)
        except:
            station_df = None
        
        return station_df
    
    def multi_gage_download(self, *gages):
        """
        Takes text files containing the gages you want to download, finds the 
        intersection of the gages with all desired parameters, downloads
        the data from each gage, and returns a dictionary of dataframes indexed 
        by gauge id.
        
        Args:
            -- *gages (str) [req]: one or multiple filepaths to text files containing gauges to download
            -- start (str)  [req]: starting download date as a string in the format '%Y-%m-%d'
            -- end (str)  [req]: ending download date as a string in the format '%Y-%m-%d'
        
        Returns:
            -- data_dict (dict): A dictionary of downloaded data indexed by gauge id
        """
        data_dict = {}
        gauge_codes_lists = {}
          
        # Read gauge IDs for each parameter from provided text files
        for file in gages:
            param = os.path.splitext(os.path.basename(file))[0]  # Extract parameter name from file name
            gauge_codes = pd.read_csv(os.path.join(self.data_dir, file), delimiter='\t', skiprows=17)
            gauge_codes = gauge_codes.iloc[1:, :]  # drop first row, metadata
            gauge_codes_lists[param] = set(gauge_codes['site_no'])
        
        # Find the intersection of gauges that have all necessary parameters
        gauges_to_download = set.intersection(*gauge_codes_lists.values())
        print(f"Gauges to Download: {len(gauges_to_download)}")
        # Download data from each gauge
        for k in gauges_to_download:
            df = self.grab_usgs_data(k, dt.datetime.strptime('1980-01-01', "%Y-%m-%d"), dt.datetime.strptime('2022-12-31', "%Y-%m-%d"))
            if df is not None:
                df['STAID'] = k
            data_dict[k] = df
        
        return data_dict

    def remove_outliers(self, arr, all_data, lower=True):
        """
        Replaces outliers above 75th percentile+1.5*IQR and below 25th percentile-1.5*IQR
        with NaNs or below 0, depending on the variable.
        
        Args:
            -- arr (arr) [req]: an array or array-like, data to be cleaned (from a single site)
            -- all_data (list) [req]: a list of data from all sites, used to calulate IQR
            -- lower (bool): Boolean to determine whether or not to clean lower boundary
        
        returns:
            -- arr (arr): cleaned array
        """
        q75, q25 = np.nanpercentile(all_data, [75, 25]) 
        iqr = q75 - q25
        upper_threshold = q75 + 1.5 * iqr
        lower_threshold = q25 - 1.5 * iqr
        arr = np.where(arr < upper_threshold, arr, np.nan)
        if lower:
            arr = np.where(arr > lower_threshold, arr, np.nan)
        else: 
            arr = np.where(arr >= 0, arr, np.nan)
        return arr
    
    def clean_data(self, data_dict):
        """
        Finds all dataframes in the data dictionary that contain all parameters,
        selects them and applies the remove outlier function to them. 
        
        Args:
            -- data_dict (dict) [req]: Dictionary containing dataframes indexed by station ID
        
        Returns:
            -- data_dict (dict): Cleaned version of data dictionary
        """
        do_list = []
        temp_list = []
        stage_list = []
        final_df_list = []
        final_gauge_list = []

        for k, v in data_dict.items():
            if v is not None:
                if all(col in v.columns for col in ['temp', 'stage', 'o']) and v['o'].count() > 100:
                    do_list.extend(v['o'].tolist())
                    temp_list.extend(v['temp'].tolist())
                    stage_list.extend(v['stage'].tolist())
                    final_df_list.append(v)
                    final_gauge_list.append(k)

        data_dict = {gauge: df for gauge, df in zip(final_gauge_list, final_df_list)}

        for k, v in data_dict.items():
            v['o'] = self.remove_outliers(v['o'], do_list, lower=False)
            v['temp'] = self.remove_outliers(v['temp'], temp_list)
            v['stage'] = self.remove_outliers(v['stage'], stage_list, lower=False)

        return data_dict
    
    def get_catchmentAttributes(self, data_dict):
        """
        Gets static catchment attributes from the NLDI API and appends them to the data dict
        """
        self.final_gauge_dict = {k: v for k, v in data_dict.items()}
        
        for k, v in data_dict.items():
            lat = data_dict[k]['lat'][0]
            lon = data_dict[k]['lon'][0]
            comid = int(NLDI().comid_byloc((lon, lat))['comid'])
            chars = NLDI().getcharacteristic_byid(comid, 'tot')
            
            # drop '_TOT' from all columns in characteristics
            chars.rename(columns=lambda x: x[4:], inplace=True)
            
            # Add static attributes to time series data
            data_arr = np.tile(chars.values, (len(v), 1))
            CA_df = pd.DataFrame(data_arr, columns=chars.columns)
            
            for column in CA_df.columns:
                col = CA_df[column].reset_index(drop=True)
                col.index =  v.index
                v[column] = col
                
            self.final_gauge_dict[k] = v
            
        return self.final_gauge_dict

        
    def extract_gagesii_data(self, data_dict, gages_II_dir):
        """
        Finds which gages are part of gages II, subsets them, and downloads 
        the desired variables from the necessary csv files. Returns a modified
        data dictionary containing USGS time-series data augmented with the 
        desired static attributes from GAGESII dataset. Note, required GAGES_II filenames
        and GAGESII variables must be stored as follows:
            data_dir + r"\GAGESII_csv_filenames.txt"
            data_dir + r'\GAGESII_vars_to_keep.txt'
        Args:
            -- data_dict (dict) [req]: Dictionary containing USGS data indexed by station ID.
            -- gages_II_dir (str) [req]: filepath to where GAGESII data is stored
        
        Returns:
            None, modifies final_gauge_dict_GII in place
        """
        print("Extracting GAGESII data")
        gagesII = pd.read_csv(gages_II_dir + r'\conterm_basinid.txt')
        gagesII_stations = list(gagesII["STAID"])

        gII_gauges = []
        final_df_list_GII = []
        for k, v in data_dict.items():
            if int(k) in gagesII_stations:
                gII_gauges.append(k)
                final_df_list_GII.append(v)
        self.final_gauge_dict = {k: v for k, v in data_dict.items() if k in gII_gauges}
        csv_filenames = pd.read_csv(self.data_dir + r"\GAGESII_csv_filenames.txt", header=None)
        csv_filenames = list(csv_filenames.iloc[:, 0])
        vars_to_grab = pd.read_csv(self.data_dir + r'\GAGESII_vars_to_keep.txt', header=None)
        vars_to_grab = list(vars_to_grab.iloc[:, 0])
        
        for k, v in self.final_gauge_dict.items():
            master_df = pd.DataFrame()
            for f in os.listdir(gages_II_dir):
                if f in csv_filenames:          
                    df = pd.read_csv(gages_II_dir + "\\" + f)
                    df = df[df["STAID"] == int(k)]
                    intersect_vars = list(set(df.columns) & set(vars_to_grab))
                    df = df[intersect_vars]
                    master_df = pd.concat([master_df, df], axis=1)

            master_df = master_df.loc[:, ~master_df.columns.duplicated()]
            data_arr = np.tile(master_df.values, (len(v), 1))
            master_df = pd.DataFrame(data_arr, columns=master_df.columns)
            # Append GAGESII data 
            for column in master_df.columns:
                col = master_df[column].reset_index(drop=True)
                col.index =  v.index
                v[column] = col
                
            self.final_gauge_dict[k] = v
            
        return self.final_gauge_dict
        
    def get_basin_geometry(self):
        """
        Gets basin geometry from the NLDI database, and stores it in a dictionary.
        Geometry is used for download DayMET, and other grid-based data sources.
 
        Returns:
            -- basin_geometry_dict (dict): dictionary containing basin geometry, indexed by station id.
        """

        basin_geometry_dict = {}
        basin_centroid_dict = {}
        for k, v in self.final_gauge_dict.items():
            polygon = NLDI().get_basins(k).geometry[0]
            basin_geometry_dict[k] = polygon
            basin_centroid_dict[k] = (polygon.centroid.x, polygon.centroid.y)

        return basin_geometry_dict, basin_centroid_dict
    
    # DayMET data download
    def download_daymet_data(self, basin_geometry_dict):
        """
        Takes basin geometry and downloads DayMET daily meteorological data for
        each basin in the final_gauge_dict. 
        
        Args:
            -- basin_geometry_dict (dict) [req]: Dictionary containing basin geometry, indexed by station ID
        Returns:
            -- final_gauge_dict (dict): Master data dictionary updated with meteorological data
        """
        last_gauge = None
        
        # Check if any data has already been downloaded and load it if it has
        intermediate_file = os.path.join(dir_path, 'final_gauge_dict.pckl')
        last_gauge_file = os.path.join(dir_path, 'last_gauge_downloaded.txt')
        if os.path.exists(intermediate_file):
            with open(intermediate_file, 'rb') as f:
                self.final_gauge_dict = pickle.load(f)
        if os.path.exists(last_gauge_file):
            with open(last_gauge_file, 'r') as f:
                last_gauge = f.read().strip()
        
        # Start at last downloaded gauge
        
        start_processing = False if last_gauge else True
            
        cnt = 1
        for k, v in self.final_gauge_dict.items():
            
            # If data has been downloaded previously, start at the last gauge that was downloaded. Otherwise start from the beginning. 
            if not start_processing:
                if k == last_gauge:
                    start_processing = True
                else:
                    continue
            # Get length of data record
            start = v.index.min().strftime('%Y-%m-%d')
            end = v.index.max().strftime('%Y-%m-%d')
            
            # Create a list to store all annual dataframes
            met_df_list = []
            
            # Only download one year at a time
            current_year = int(start[:4])
            end_year = int(end[:4])
            
            while current_year <= end_year:
                year_start = f"{current_year}-01-01"
                year_end = f"{current_year}-12-31"
                date_range = (year_start, year_end)
                
                # Loop until data download is successful or until 1 hour is up
                returnVal = None
                start_time = time.time()
                timeout = 3600  # 1 hour in seconds
                
                while returnVal is None:
                    try:
                        if time.time() - start_time > timeout:
                            print(f"Timeout exceeded for {year_start} to {year_end} at USGS gauge {k}. Moving onto next year")
                            break
                        print(f"Trying to download {current_year} DayMET data at USGS gauge {k} \n Gauge {cnt} out of {len(basin_geometry_dict)}")
                        met_data = daymet.get_bycoords(basin_geometry_dict[k], date_range)
                        current_year += 1
                        returnVal = 1 
                    except Exception as e:
                        print("DayMET download failed, trying again")
                        print(f"Error: {str(e)}")
                        traceback.print_exc()

                met_df_list.append(met_data)
                print(met_data.head())
                
                
            met_df = pd.concat(met_df_list)
            
            # Append meteorological data to gauge data
            for column in met_df.columns:
                col = met_df[column]
                v[column] = col
                v = v.sort_index()
            
            # update master data dictionary
            self.final_gauge_dict[k] = v
            
            # Update count for keeping track of progress
            cnt += 1
            
            # Save current download state
            with open('final_gauge_dict.pckl', 'wb') as f:
                pickle.dump(self.final_gauge_dict, f)
            print("Intermediate data saved")
            with open('last_gauge_downloaded.txt', 'w') as f:
                f.write(k)
            
        return self.final_gauge_dict
    
    def combine_data(self, fname):
        """
        Combines all data together and saves as a csv to desired filename.
        
        Args:
            -- fname (str) [req]: saves csv to this location in the data directory
        Returns:
            -- None
        """
        dfs = [df for df in self.final_gauge_dict.values()]
        final_df = pd.concat(dfs)
        final_df.to_csv(self.data_dir + fname, index=True)

#%%
# Main script
# if __name__ == "__main__":
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
data_dir = dir_path + '\\data'

# Create dictionary of parameters and their respective USGS codes
param_dict = {'stage': '00060', 'temp': '00010', 'o': '00300'}

# Instatiate DataGrabber class
data_grabber = DataGrabber(data_dir, param_dict)

# Check for saved state
if os.path.exists(dir_path + r'\final_gauge_dict_GII.pckl'):
    print("exists")
    with open('final_gauge_dict_GII.pckl', 'rb') as f:
        data_grabber.final_gauge_dict_GII = pickle.load(f)
    data_grabber.data_dir = data_dir
    data_grabber.param_dict = param_dict
else:
    print("Unable to load data")
    
# Get USGS data
gages_files = ['DO_usgs_gauges.txt', 'stage_usgs_gauges.txt', 'temp_usgs_gauges.txt']
# gages_files = ['test_gauge.txt']
stream_data_dict = data_grabber.multi_gage_download(*gages_files)
    
print("USGS stream data downloaded, grabbing USGS lake data")

# Get Lakes data
lake_files = ['DO_usgs_lakes.txt', 'stage_usgs_lakes.txt', 'temp_usgs_lakes.txt']
lake_data_dict = data_grabber.multi_gage_download(*lake_files)

# Combine stream and lake data into a master data dictionary
master_data_dict = stream_data_dict.update(lake_data_dict)

#%%
data_dict_cleaned = data_grabber.clean_data(master_data_dict)
HMCA_data = data_grabber.get_catchmentAttributes(data_dict_cleaned)

#%% Manual HMCA extraction
HMCA_data = {k: v for k, v in data_dict_cleaned.items()}

for k, v in data_dict_cleaned.items():
    lat = data_dict_cleaned[k]['lat'][0]
    lon = data_dict_cleaned[k]['lon'][0]
    try:
        comid = int(NLDI().comid_byloc((lon, lat))['comid'])
        chars = NLDI().getcharacteristic_byid(comid, 'tot')
        
        # drop '_TOT' from all columns in characteristics
        chars.rename(columns=lambda x: x[4:], inplace=True)
        
        # Add static attributes to time series data
        data_arr = np.tile(chars.values, (len(v), 1))
        CA_df = pd.DataFrame(data_arr, columns=chars.columns)
        
        for column in CA_df.columns:
            col = CA_df[column].reset_index(drop=True)
            col.index =  v.index
            v[column] = col
        HMCA_data[k] = v
    except:
        HMCA_data[k] = None
        

#%% DayMET Data
print("Getting basin geometry")
basin_geometry_dict, basin_centroid_dict = data_grabber.get_basin_geometry()
if basin_geometry_dict:
    master_data_dict = data_grabber.download_daymet_data(basin_centroid_dict)
print("DayMet download successful. Combining data into one CSV.")
        
#%% Daymet
master_data_dict = data_grabber.download_daymet_data(basin_centroid_dict)

#%% Combine data into one csv and save
fname = data_dir + '\\training_data_115GII_gauges.csv'
data_grabber.combine_data(fname)


#%% read in data from pickle
with open(dir_path+'/final_gauge_dict_temp.pckl', 'rb') as f:
    HMCA_data = pickle.load(f)

#%% Set class and final gauge dict manually
param_dict = {'stage': '00060', 'temp': '00010', 'o': '00300'}
data_grabber = DataGrabber(data_dir, param_dict)
data_grabber.final_gauge_dict = HMCA_data

#%% manual basin geometry
basin_geometry_dict = {}
basin_centroid_dict = {}
for k, v in HMCA_data.items():
    try:
        polygon = NLDI().get_basins(k).geometry[0]
        basin_geometry_dict[k] = polygon
        basin_centroid_dict[k] = (polygon.centroid.x, polygon.centroid.y)
    except:
        basin_geometry_dict[k] = None
        basin_centroid_dict[k] = None
        
        #%% Filter out basins with no geometry
basin_centroid_dict = {k: v for k, v in basin_centroid_dict.items() if v is not None}
HMCA_filtered = {k: v for k, v in HMCA_data.items() if k in basin_centroid_dict}
#%% Filter out basins with no hydro/WQ data
HMCA_filtered = {k: v for k, v in HMCA_filtered.items() if v is not None}
#%% Only download sites that haven't yet been downloaded
HMCA_filtered = {k: v for k, v in HMCA_filtered.items() if len(v.columns) == 133}

# QC check
# counter = 0
# for k, v in HMCA_filtered.items():
#     if v is not None:
#         if len(v.columns) == 140:
#             counter += 1
            
# print(counter)

#%% Manual Daymet
cnt = 1
for k, v in HMCA_filtered.items():

    # Get length of data record
    start = v.index.min().strftime('%Y-%m-%d')
    end = v.index.max().strftime('%Y-%m-%d')
    
    # Create a list to store all annual dataframes
    met_df_list = []
    
    # Only download one year at a time
    current_year = int(start[:4])
    end_year = int(end[:4])
    
    while current_year <= end_year:
        year_start = f"{current_year}-01-01"
        year_end = f"{current_year}-12-31"
        date_range = (year_start, year_end)
        
        # Loop until data download is successful or until 1 hour is up
        returnVal = None
        start_time = time.time()
        timeout = 3600  # 1 hour in seconds
        
        while returnVal is None:
            try:
                if time.time() - start_time > timeout:
                    print(f"Timeout exceeded for {year_start} to {year_end} at USGS gauge {k}. Moving onto next year")
                    break
                print(f"Trying to download {current_year} DayMET data at USGS gauge {k} \n Gauge {cnt} out of {len(HMCA_filtered)}")
                # met_data = daymet.get_bycoords(basin_geometry_dict[k], date_range, variables=var)
                
                met_data = daymet.get_bycoords(basin_centroid_dict[k], date_range)

                # met_data = daymet.get_bystac(basin_geometry_dict[k], date_range, variables=var)
                current_year += 1
                returnVal = 1 

            except Exception as e:
                print("DayMET download failed, trying again")
                print(f"Error: {str(e)}")
                traceback.print_exc()

        met_df_list.append(met_data)
        print(met_data.columns)
        print(met_data.head())
        
        
    met_df = pd.concat(met_df_list)
    # Append meteorological data to gauge data
    for column in met_df.columns:
        col = met_df[column]
        v[column] = col
        v = v.sort_index()
    
    # update master data dictionary
    HMCA_data[k] = v
    
    # Update count for keeping track of progress
    cnt += 1

#%%
# Save current download state
with open('final_gauge_dict_temp.pckl', 'wb') as f:
    pickle.dump(HMCA_data, f)
print("Intermediate data saved")
with open('last_gauge_downloaded.txt', 'w') as f:
    f.write(k)
    
#%% QC Check
counter = 0
for k, v in HMCA_data.items():
    if v is not None:
        if len(v.columns) == 140:
            counter += 1
            
print(counter)

#%% Testing
gages_files = ['test_gauge.txt']
data_grabber_test = DataGrabber(data_dir, param_dict)
data_dict = data_grabber_test.multi_gage_download(*gages_files)
test_gauge = ['01400500']
lat, lon = data_grabber_test.get_gage_location(test_gauge[0])
geometry = NLDI().get_basins(test_gauge).geometry[0]

comid = int(NLDI().comid_byloc((lon, lat))['comid'])
chars = NLDI().getcharacteristic_byid(comid, 'tot')

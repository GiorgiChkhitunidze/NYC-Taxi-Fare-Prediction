# imports
import pandas as pd
import numpy as np
from datetime import date
from matplotlib.dates import date2num, num2date
import folium
import geopandas as gpd
import json
from shapely.geometry import Point, shape
from sklearn.metrics import r2_score
from tqdm.notebook import  tqdm

def adjusted_r2(y_true, y_pred, k):
    """
    Calculate Adjusted R-Squared

    ## Parameters
    y_true: True values

    y_pred: Predicted values

    k: Number of predictors
    """

    # Calculate R-squared score
    r2 = r2_score(y_true, y_pred)

    # Calculate adjusted R-squared
    n = len(y_true)
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - k - 1))

    print("Adjusted R-squared:", round(adjusted_r2, 5))



def check_points_inside_valid_range(geojson_file, df, longitude, latitude, new_col_name):
    """
    Gets DataFrame that contains coordinates and return whether they are inside or outside valid range

    ## Parameters
    geojson_file: GeoJson file that contains polygon surrounding valid coordinates

    df: DataFrame containing coordinates that need to be checked

    longitude: Column name of longitude coordinates

    latitude: Column name of latitude coordinates

    new_col_name: New column name
    """

    # Load the GeoJSON file
    with open(geojson_file) as f:
        geojson_data = json.load(f)

    # Create the valid range polygon shape
    polyg_shape = shape(geojson_data['features'][0]['geometry'])

    # Create a shapely point object for each coordinate in the DataFrame
    print('Creating shapely points...')
    points = [Point(row[longitude], row[latitude]) for _, row in tqdm(df.iterrows(), total=df.shape[0], position=0, leave=True)]

    # Use vectorized contains operation to check if points are inside the valid range
    print('Checking points...')
    inside_mask = [polyg_shape.contains(point) for point in tqdm(points, total=df.shape[0], position=0, leave=True)]

    # Add a new column indicating whether each point is inside or outside the country
    print('Returning DataFrame...')
    df[new_col_name] = pd.Series(inside_mask, index=df.index).map({True: 'Inside', False: 'Outside'})
    print('Done!')

    return df



def get_borough_names(df):
    """
    Get the borough names that the coordinates belong to
    NOTE: if location is not in any borough it will be filled with "Outside NYC"

    ## Parameters
    df: Dataframe that contains coordinates
    """
    # Load the GeoJSON file
    print('Loading borough boundaries file...')
    nyc_boroughs = gpd.read_file('boroughs/Borough Boundaries.geojson')

    # Get coordinates
    print('Getting pickup and dropoff coordinates...')
    print(' - Pickup')
    pickup_longitudes = df['pickup_longitude'].values
    pickup_latitude = df['pickup_latitude'].values

    print(' - Dropoff')
    dropoff_longitude = df['dropoff_longitude'].values
    dropoff_latitude = df['dropoff_latitude'].values


    # Create a GeoDataFrame with the coordinates
    print('Creating GeoDataFrames with the coordinates...')
    print(' - Pickup')
    pickup_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(pickup_longitudes, pickup_latitude))
    print(' - Dropoff')
    dropoff_points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(dropoff_longitude, dropoff_latitude))

    # Perform spatial join and extract borough name
    print('Performing spatial join and extract borough name...')
    print(' - Pickup')
    pickup_boro = gpd.sjoin(pickup_points, nyc_boroughs, how='left', op='intersects')['boro_name']
    print(' - Dropoff')
    dropoff_boro = gpd.sjoin(dropoff_points, nyc_boroughs, how='left', op='intersects')['boro_name']

    # add borougs to df
    print('Adding borough names to df...')
    df['pickup_boro_name'] = pickup_boro.values
    df['pickup_boro_name'].fillna('Outside NYC', inplace=True)
    df['dropoff_boro_name'] = dropoff_boro.values
    df['dropoff_boro_name'].fillna('Outside NYC', inplace=True)
    print('Done!')



# airports dataframe
airports_df = pd.read_csv('boroughs/Airports.csv')
airports_df['latitude'] = airports_df.WKT.str.split().str[2].str[:-1].astype('float')
airports_df['longitude'] = airports_df.WKT.str.split().str[1].str[1:].astype('float')

### Style for polygon around NYC
def style_function(feature):
    return {
        'fillColor': 'orange',  # Set the fill color of the polygon
        'color': 'orange',      # Set the border color of the polygon
        'weight': 2,             # Set the border weight of the polygon
        'fillOpacity': 0.1,      # Set the opacity of the polygon
    }


# Draw a map at the location of NYC
def folium_NYC(zoom_start = 10, tiles = 'cartodbpositron', add_polygon = False):
    """
    Draw folium map. Initialize location at NYC.

    ## Parameters
    zoom_start: Initial zoom level when map loads (default: 10)

    tiles: Map tileset or map style (default: cartodbpositron)

    add_polygon: Add polygon around NYC that can would surround valid taxi coordinates
    """

    # initialize folium graph at NYC location
    m = folium.Map(location = [40.730610, -73.935242], zoom_start = zoom_start, tiles = tiles)

    if add_polygon==True:
        geo_json_polygon = json.load(open('boroughs\Around_NYC.geojson'))
        folium.GeoJson(geo_json_polygon, 
                       style_function=style_function,
                       name='<span style="color: orange;">Valid Taxi Ride Range</span>').add_to(m)

    # Mark each Boroughs in different colors
    geo_json_map = json.load(open('boroughs/Borough Boundaries.geojson'))
    for idx, color in enumerate(['red', 'purple', 'yellow', 'green', 'Blue']):
        coords = geo_json_map['features'][idx]['geometry']['coordinates']
        boro = geo_json_map['features'][idx]['properties']['boro_name']
        lgd_txt = '<span style="color: {col};">{txt}</span>'

        fg = folium.FeatureGroup(name = lgd_txt.format(txt = boro, col = color))
        pl = folium.Polygon(
            [
                list(
                    zip(
                        np.array(coords[i][0])[:,1],
                        np.array(coords[i][0])[:,0]
                    )
                ) for i in range(len(coords))
            ],
            color=color,
            weight=1, 
            opacity=0.8,
            fill_color = color,
            fill_opacity =0.2
        ).add_to(fg)
        m.add_child(fg)
    folium.map.LayerControl('topleft', collapsed= False).add_to(m)


    return m



# add taxi pickup/dropoff locations on after drawing a map
def add_pickup_dropoff_markers(m, pickup_coords:list=None, dropoff_coords:list=None):
    """
    Add pickup/dropoff locations on the map
    NOTE: First you have to draw map using folium_NYC() function.

    ## Parameters
    m: Map returned by folium_NYC() funtion

    pickup_coords: List of taxi pickup coordinates (eg. [40.730610, -73.935242])

    dropoff_coords: List of taxi dropoff coordinates (eg. [40.730610, -73.935242])
    """
    if pickup_coords!=None or dropoff_coords!=None:
        # Case 1: show both pickup and dropoff locations
        if pickup_coords!=None and dropoff_coords!=None:
            for i in range(len(pickup_coords)):
                # Pickup locations
                folium.Marker(
                    location = pickup_coords[i],
                    popup = f'Pickup ({pickup_coords[i][0]}, {pickup_coords[i][1]})\
                        Dropoff ({dropoff_coords[i][0]}, {dropoff_coords[i][1]})', 
                    icon=folium.Icon(
                        color='red', 
                        prefix='fa', 
                        icon='taxi'
                    )
                ).add_to(m)

                # Dropoff locations
                folium.Marker(
                    location = dropoff_coords[i], 
                    popup = f'Pickup ({pickup_coords[i][0]}, {pickup_coords[i][1]})\
                        Dropoff ({dropoff_coords[i][0]}, {dropoff_coords[i][1]})', 
                    icon=folium.Icon(
                        color='blue', 
                        prefix='fa', 
                        icon='taxi'
                    )
                ).add_to(m)

        # Case 2: Show only pickup coordinates
        elif pickup_coords!=None:
            for i in range(len(pickup_coords)):
                folium.Marker(
                    location = pickup_coords[i],
                    popup = f'Pickup ({pickup_coords[i][0]}, {pickup_coords[i][1]})', 
                    icon=folium.Icon(
                        color='red', 
                        prefix='fa', 
                        icon='taxi'
                    )
                ).add_to(m)

            # Case 3: Show only dropoff coordinates 
        else:
            for i in range(len(dropoff_coords)):
                folium.Marker(
                    location = dropoff_coords[i], 
                    popup = f'Dropoff ({dropoff_coords[i][0]}, {dropoff_coords[i][1]})', 
                    icon=folium.Icon(
                        color='blue', 
                        prefix='fa', 
                        icon='taxi'
                    )
                ).add_to(m)
    return m


# Add airport locations after drawing a map
def add_airport_locations(m):
    """
    Add airport locations on the map
    NOTE: First you have to draw map using folium_NYC() function.

    ## Parameters
    m: Map returned by folium_NYC() funtion
    """
    for idx in list(airports_df.index):
        folium.Marker(
            location = [
                airports_df.loc[idx, 'latitude'],
                airports_df.loc[idx, 'longitude']
                ],
            popup = airports_df.loc[idx, 'name'],
            icon=folium.Icon(
                color='green',
                prefix='fa',
                icon='plane')
        ).add_to(m)
        
    return m


# Calculate distance covered during a ride
def distance_covered(df, df_name:str):
    """
    Calculate distance covered during a ride

    ## Parameters
    df: DataFrame in which the pickup/dropoff latitudes and longitudes reside

    df_name:  Variable name of DataFrame. Will be used to track progress
    """
    print(f'Calculating distance covered for "{df_name}"...')
    print('Converting latitude and longitude from degrees to radians...')
    # Convert latitude and longitude from degrees to radians
    lat1_rad = np.radians(df.pickup_latitude.values)
    lon1_rad = np.radians(df.pickup_longitude.values)
    lat2_rad = np.radians(df.dropoff_latitude.values)
    lon2_rad = np.radians(df.dropoff_longitude.values)

    # Radius of the Earth in kilometers
    radius = 6371.0

    # Haversine formula
    print('Mesuring distance...')
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = radius * c

    print(f'Adding "distance_covered" to {df_name}...')
    df['distance_covered'] = distance
    print('Done!', end='\n\n')


# Haversine function for airport_distance formula below
def haversine_formula(lat, long, ariport_lat, ariport_long):
    """
    Haversine formula to calculate distance between two coordinates

    ## Parameters
    lat: Latitude of pickup/dropoff coordinate

    long_name:  Longitude of pickup/dropoff coordinate

    ariport_lat: Latitude of airport location

    ariport_long: Longitude of airport location
    """
    # Radius of the Earth in kilometers
    radius = 6371.0

    # Haversine formula
    dlat = lat - ariport_lat
    dlon = long - ariport_long

    a = np.sin(dlat/2)**2 + np.cos(ariport_lat) * np.cos(lat) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    distance = radius * c
    return distance



# Calculate distance to given airpirt
def airport_distance(df, df_name, airport_name:str):
    """
    Calculate distance from pickup/dropoff location to airport

    ## Parameters
    df: DataFrame in which the pickup/dropoff latitudes and longitudes reside

    df_name:  Variable name of DataFrame. Will be used to track progress

    airport_name: String value of airport name
    """
    print(f'Calculating distance to "{airport_name}" for "{df_name}"...')

    # Convert latitude and longitude from degrees to radians
    print('Converting latitude and longitude from degrees to radians...')
    pickup_lat = np.radians(df.pickup_latitude.values)
    pickup_long = np.radians(df.pickup_longitude.values)
    dropoff_lat = np.radians(df.dropoff_latitude.values)
    dropoff_long = np.radians(df.dropoff_longitude.values)
    ariport_lat = np.radians(airports_df.loc[airports_df.name==airport_name, 'latitude'].values)
    ariport_long = np.radians(airports_df.loc[airports_df.name==airport_name, 'longitude'].values)

    # Measure distance
    print(f'Measuring distance between pickup location and "{airport_name}"...')
    pickup_airport_distance = haversine_formula(pickup_lat, pickup_long, ariport_lat, ariport_long)
    print(f'Measuring distance between dropoff location and "{airport_name}"...')
    dropoff_airport_distance = haversine_formula(dropoff_lat, dropoff_long, ariport_lat, ariport_long)
 
    # Add to datafrme
    print(f'Adding "pickup_{airport_name}_distance" to {df_name}...')
    df[f'pickup_{airport_name}_distance'] = pickup_airport_distance

    print(f'Adding "dropoff_{airport_name}_distance" to {df_name}...')
    df[f'dropoff_{airport_name}_distance'] = dropoff_airport_distance

    print('Done!', end='\n\n')


# extrcat date features such as date, year, month, week, day, hour
def get_date_features(df, df_name:str):
    """
    Extract date features. Give DataFrame and the function will add date, year, month, week, day, hour columns

    ## Parameters
    df: DataFrame in which column with DateTime format resides

    df_name:  Variable name of DataFrame. Will be used to track progress
    """
    print(f'Etracting features from "{df_name}"...')

    # convert to datetime
    print('Converting "pickup_datetime" from string to datetime...')
    df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
    print('Done!')

    # extract features
    print('Extracting "date"...')
    df['date'] = df['pickup_datetime'].dt.date
    print('Done!')
    print('Extracting "year"...')
    df['year'] = df['pickup_datetime'].dt.year
    print('Done!')
    print('Extracting "month"...')
    df['month'] = df['pickup_datetime'].dt.month
    print('Done!')
    print('Extracting "week of a year"...')
    df['week'] = df["pickup_datetime"].dt.isocalendar().week.astype('float64')
    print('Done!')
    print('Extracting "day of a month"...')
    df['day_of_month'] = df['pickup_datetime'].dt.day
    print('Done!')
    print('Extracting "day of a week"...')
    df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
    print('Done!')
    print('Extracting "hour of a day"...')
    df['hour'] = df['pickup_datetime'].dt.hour
    print('Done!', end='\n\n')


# Assign 0 to observation below November of 2012 and 1 to above
def get_numeric_date_aft_nov_2012(df, df_name:str):
    """
    Extract date features. Give DataFrame and the function will add date, year, month, week, day, hour columns

    ## Parameters
    df: DataFrame in which Date column resides

    df_name:  Variable name of DataFrame. Will be used to track progress
    """
    # date to numeric date
    print(f'Working On DataFrame: "{df_name}"')
    print('Converting "date" to numeric and adding to "numeric_date" column...')
    df['numeric_date'] = df['date'].apply(date2num)
    print('Done!')

    # add after_nov_2012: 0:before November of 2012, 1: after November of 2012
    print(f'Adding "after_nov_2012" colum...')
    df['after_nov_2012'] = 1
    df.loc[df['date'] <= date(2012, 9, 3), 'after_nov_2012'] = 0
    print('Done!', end='\n\n')



def remove_outliers_iqr(column_name, train_df, valid_df=None, test_df=None):
    """
    Remove outliers using Interquartile Range method

    ## Parameters
    column_name: Column from which we want to remove outliers

    train_df: dataframe that contains training data

    valid_df: dataframe that contains validation data

    test_df: dataframe that contains testing data
    """
    print('Calculating valid ranges from "Training data"...')
    # Calculate the IQR
    q1 = train_df[column_name].quantile(0.25)
    q3 = train_df[column_name].quantile(0.75)
    iqr = q3 - q1
    
    # Define the lower and upper bounds
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Remove outliers
    print(f'Removing outliers from "Training data":"{column_name}"...')
    before_len = train_df.shape[0]
    train_df = train_df[(train_df[column_name] >= lower_bound) & (train_df[column_name] <= upper_bound)]
    print(f"{before_len-train_df.shape[0]} observations removed!")
    print(f"{train_df.shape[0]} observations remain!", end='\n\n')

    if valid_df is not None:
        print(f'Removing outliers from "Validation data":"{column_name}"...')
        before_len = valid_df.shape[0]
        valid_df = valid_df[(valid_df[column_name] >= lower_bound) & (valid_df[column_name] <= upper_bound)]
        print(f"{before_len-valid_df.shape[0]} observations removed!")
        print(f"{valid_df.shape[0]} observations remain!", end='\n\n')
    
    if test_df is not None:
        print(f'Removing outliers from "Testing data":"{column_name}"...')
        before_len = test_df.shape[0]
        test_df = test_df[(test_df[column_name] >= lower_bound) & (test_df[column_name] <= upper_bound)]
        print(f"{before_len-test_df.shape[0]} observations removed!")
        print(f"{test_df.shape[0]} observations remain!", end='\n\n')
    
    if (valid_df is not None) and (test_df is not None):
        return train_df, valid_df, test_df
    
    elif (test_df is not None):
        return train_df, test_df
    
    elif (valid_df is not None):
        return train_df, valid_df
    else:
        return train_df
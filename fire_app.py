from pandas.io.parsers import read_csv
import streamlit as st
import sklearn
import numpy as np
import pandas as pd
import datetime
import plotly.express as px
import plotly
import plotly.graph_objects as go
from scipy.spatial import distance
import xgboost as xgb
from joblib import dump, load
from shapely.geometry import MultiPoint
import geopandas as gpd

@st.cache
def load_data():
    """Loads relevant data from csv files

    Returns:
        pd.DataFrame: green light, dfd locations, and incident data frames
    """
    green_light = pd.read_csv(r"Project_Green_Light_Locations.csv")
    green_light['live_date'] = pd.to_datetime(green_light['live_date'])
    dfd_locations = pd.read_csv(r"DFD_Fire_Station_Locations.csv")
    fire_inc = pd.read_csv(r"Fire_Incidents_Transformed.csv")

    return green_light, dfd_locations, fire_inc

@st.cache
def make_prediction(in_date, in_time, in_lat, in_lon, green_lights, dfd_locs):
    """Generates probability of injury/fatality baed on input parameters

    Args:
        in_date (datetime.date): prediction date
        in_time (datetime.time): prediction time
        in_lat (float): prediction latitude
        in_lon (float): prediction longitude
        green_lights (np.Array): green light coordinates
        dfd_locs (np.Array): dfd coordinates

    Returns:
        float: probability of injury/fatality
    """

    ##extract date parts from input
    dtime = datetime.datetime.combine(in_date, in_time)
    dtime_struct = dtime.timetuple()
    hour = dtime_struct.tm_hour
    day = dtime_struct.tm_mday
    doy = dtime_struct.tm_yday
    dow = dtime_struct.tm_wday
    week = dtime.isocalendar()[1]
    month = dtime_struct.tm_mon

    ##transform date parts using sin/cos functions
    hour_x, hour_y = get_hour_transform(hour)
    day_x, day_y = get_day_transform(day)
    DoY_x, DoY_y = get_doy_transform(doy)
    DoW_x, DoW_y = get_dow_transform(dow)
    week_x, week_y = get_week_transform(week)
    month_x, month_y = get_month_transform(month)

    ##get distance to nearest station and green light
    closest_light, clostes_stn = get_distances(in_lat, in_lon, green_lights, dfd_locs)

    ##get location cluster
    cluster = get_cluster(in_lat, in_lon)
    pred_df = pd.DataFrame(data=
        {'hourx':[hour_x],
        'houry':[hour_y],
        'dayx':[day_x],
        'dayy':[day_y],
        'DoYx':[DoY_x],
        'DoYy':[DoY_y],
        'DoWx':[DoW_x],
        'DoWy':[DoW_y],
        'weekx':[week_x],
        'weeky':[week_y],
        'monthx':[month_x],
        'monthy':[month_y],
        'closest_light':[closest_light],
        'closest_stn':[clostes_stn],
        })

    

    model = xgb.XGBClassifier()
    model.load_model("boost_model.json")

    model_features = ['hourx','houry','dayx', 'dayy', 'DoYx', 'DoYy', 'DoWx',
        'DoWy', 'weekx', 'weeky', 'monthx', 'monthy', 'closest_stn',
        'closest_light', 'cluster_7', 'cluster_20', 'cluster_23', 'cluster_21',
        'cluster_0', 'cluster_15', 'cluster_19', 'cluster_5', 'cluster_9',
        'cluster_8', 'cluster_2', 'cluster_17', 'cluster_10', 'cluster_3',
        'cluster_22', 'cluster_14', 'cluster_11', 'cluster_13', 'cluster_18',
        'cluster_24', 'cluster_1', 'cluster_4', 'cluster_16', 'cluster_6',
        'cluster_12']

    ##create indicator vars for cluster columns
    cluster_cols = [x for x in model_features if "cluster" in x]
    pred_df[cluster_cols] = 0
    cluster_col = "cluster_" + str(cluster)
    pred_df[cluster_col] = 1

    pred_df = pred_df[model_features] ##reorder columns

    prediction = model.predict_proba(pred_df)[0,1] ##row 0, col 1

    return round(prediction, 2)

@st.cache
def get_cluster(lat, lon):
    """Returns cluster number of latitude/longitude input

    Args:
        lat (float): latitude
        lon (float): longitude

    Returns:
        int: cluster #
    """
    clustering = load('cluster_model.joblib')   # long = x, lat = y
    return clustering.predict([[lat, lon]])[0]

def get_distances(lat, lon, light_coords, dfd_coords):
    """calculates the nearest green light location and dfd station

    Args:
        lat (float): latitude
        lon (float): longitude
        light_coords (np.Array): green light coordinates
        dfd_coords (np.Array): dfd oordinates

    Returns:
        float: distance from input location to nearest light and station
    """
    
    light_dist = distance.cdist(
                            np.array([(lon, lat)]),
                            light_coords).min(axis=1)[0]
    stn_dist = distance.cdist(
                            np.array([(lon, lat)]),
                            dfd_coords).min(axis=1)[0]
    
    return  light_dist, stn_dist

def get_hour_transform(hour):
    hour_x = np.sin(2 * np.pi * hour / hour_periods())
    hour_y = np.cos(2 * np.pi * hour / hour_periods())

    return hour_x, hour_y

def get_day_transform(day):
    day_x = np.sin(2 * np.pi * day / day_periods())
    day_y = np.cos(2 * np.pi * day / day_periods())

    return day_x, day_y

def get_doy_transform(doy):
    doy_x = np.sin(2 * np.pi * doy / doy_periods())
    doy_y = np.cos(2 * np.pi * doy / doy_periods())

    return doy_x, doy_y

def get_dow_transform(dow):
    dow_x = np.sin(2 * np.pi * dow / dow_periods())
    dow_y = np.cos(2 * np.pi * dow / dow_periods())

    return dow_x, dow_y

def get_week_transform(week):
    week_x = np.sin(2 * np.pi * week / week_periods())
    week_y = np.cos(2 * np.pi * week / week_periods())

    return week_x, week_y

def get_month_transform(month):
    month_x = np.sin(2 * np.pi * month / month_periods())
    month_y = np.cos(2 * np.pi * month / month_periods())

    return month_x, month_y

def hour_periods():
    return 24

def day_periods():
    return 31

def doy_periods():
    return 365

def dow_periods():
    return 7

def week_periods():
    return 53

def month_periods():
    return 12

def hash_return_1():
    return 1

@st.cache
def get_heatmap(fire_inc, x, y):
    """generates heatmap of incident counts based on x/y columns 

    Args:
        fire_inc (pd.DataFrame): fire incidents
        x (str): column in fire_inc that will be on x axis of heatmap
        y (str): column in fire_inc that will be on y axis of heatmap

    Returns:
        plotly.graph_object: heatmap graph object
    """
    counts = fire_inc.groupby(by=[x, y])['injury_or_fatality'].count().reset_index()
    counts = counts.pivot(index=x, columns=y, values='injury_or_fatality')
    return px.imshow(counts)

def get_geo_df(fire_inc):
    """generate geopandas dataframe with cluster details

    Args:
        fire_inc (pd.DataFrame): fire incidents

    Returns:
        gpd.GeoDataFrame: dataframe with cluster geometries
    """
    clusters = {'cluster': [], 'geometry': [], 'incident_count': [],
        'inj_per_inc': [], 'inj': []}
    for cluster in fire_inc['cluster'].unique():
        if cluster == -1:
            continue
        points = fire_inc[fire_inc['cluster'] == cluster]
        clusterPoints = MultiPoint(list(zip(points['x'], points['y'])))
        clusterPoly = clusterPoints.convex_hull
        clusters['cluster'].append(cluster)
        clusters['geometry'].append(clusterPoly)
        clusters['incident_count'].append(len(points))
        clusters['inj_per_inc'].append(
                                len(points[points['injury_or_fatality'] == 1])/len(points)*1000)
        clusters['inj'].append(len(points[points['injury_or_fatality'] == 1]))

    geoDf = gpd.GeoDataFrame(clusters, crs='EPSG:4326')

    return geoDf

@st.cache
def get_cluster_map(fire_inc, dfd_locations):
    """generate figure to display incident clusters w/ injury rate

    Args:
        fire_inc (pd.DataFrame): fire incidents
        dfd_locations (pd.DataFrame): dfd locations

    Returns:
        plotly.graph_object: [map figure
    """
    geoDf = get_geo_df(fire_inc)

    fig = go.Figure()
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geoDf.__geo_interface__,
            locations=geoDf.cluster,
            featureidkey="properties.cluster",
            ids=geoDf.index,
            z=geoDf.inj_per_inc,
            colorscale=[(0,"white"), (1,"red")],
            marker={'opacity':.6},
            name='Incident Cluster',
        )
    )
    fig.add_trace(
        go.Scattermapbox(
            lat=dfd_locations['Y'],
            lon=dfd_locations['X'],
            marker={'size':5, 'color':'black'},
            name='DFD Location'
        )
    )
    fig.update_layout(
        mapbox=dict(
            bearing=0,
            center = {"lat": 42.38, "lon": -83.08},
            pitch=0,
            zoom=9,
            style='light'
        ),
        mapbox_style="open-street-map",
        title="Injury/Fatality Rate per 1K Incidents"
    )

    return fig

@st.cache
def display_hour_transform(df):
    """Generates figure of transformed hour feature thorugh sin/cos

    Args:
        df (pd.DataFrame): dataframe of fire incidents

    Returns:
        plotly.graph_object: scatter plot
    """

    ##apply transform to hour column
    df['hour_x'] = df['hour'].map(lambda x: get_hour_transform(x)[0]).round(2)
    df['hour_y'] = df['hour'].map(lambda x: get_hour_transform(x)[1]).round(2)
    
    ##aggregrate the results so we can get counts
    day_counts = (df.groupby(by=['hour_x','hour_y', 'hour'])
                    .agg(**{"Incident Count": ("x", "count")})
                    .reset_index())

    fig = px.scatter(
            day_counts,
            x="hour_x",
            y="hour_y",
            hover_data=["hour"],
            color="Incident Count")

    return fig

'''
# Analysis of Fire Incidents in City of Detroit
The goal of this project has three main objectives:
- Gather interesting insights from data analysis and visualization
- Build a predictive model to determine the likelihood of injury or fatality for
a given fire incident
- Allow for real-time prediction given user input

This project collects publicly available data from the City of Detroit. Three
datasets are used and combined to create additional predictors.

1. [Fire Incidents](https://data.detroitmi.gov/datasets/fire-incidents/explore) -
this is the main dataset that contains all reported fire
incidents throughout the city. We'll look at years 2017-2019 specifically. 
1. Fire Station Locations - used to calculate the nearest fire station to each
incident.
1. Project Green Light Locations - used as a proxy for crime/safety of nearby
area. 
'''

st.sidebar.title("Enter Prediction Inputs")
in_date = st.sidebar.date_input("Date")
in_time = st.sidebar.time_input("Time", )
lat = 42.33
lon = -83.05
lat = st.sidebar.number_input("Latitude", min_value=42.3, max_value=42.45, value=lat)
lon = st.sidebar.number_input("Longitude", min_value=-83.20, max_value=-82.95, value=lon)

##add map to sidebar to show prediction location input
fig = go.Figure()
fig.add_trace(
    go.Scattermapbox(
        lat=np.array([lat]),
        lon=np.array([lon]),
        marker={'size':8, 'color':'red'},
        name='Selected Location'
    )
)
fig.update_layout(
    mapbox=dict(
        bearing=0,
        center = {"lat": lat, "lon": lon},
        pitch=0,
        zoom=9,
        style='light'
    ),
    mapbox_style="open-street-map",
    title="Prediction Location"
)
st.sidebar.plotly_chart(fig, use_container_width=True)

green_lights, dfd_locs, fire_inc = load_data()

'''
Let's take a look at our partially transformed dataset. We'll explain how we 
contruct additional feature below.
'''

st.dataframe(fire_inc.head())

'''
The Timestamp of each incident was broken out into it's respective dat parts. 
However, when training the model we can't use the raw date parts or else we lose
the cyclical nature of time. In other words, hour 0 and hour 23 are close
together in reality, by if we use the features as they are the model will treat
these values as far apart. Therefore, we need to apply sine and cosine 
transforms to each datepart. Our result is two numbers representing each date 
part as we can see below. Hours 0 and 12 are on opposite "sides", but 23 and 0 
are close together.
'''

st.plotly_chart(display_hour_transform(fire_inc.copy()))

##convert dataframes to list of coordinate tuples
dfd_coords = np.array(list(zip(dfd_locs['X'], dfd_locs['Y'])))
light_coords = np.array(list(zip(green_lights['X'], green_lights['Y'])))

##create heatmaps of incidents
st.subheader(f"# of Incidents by Day of Week and Hour")
st.plotly_chart(get_heatmap(fire_inc, 'DoW', 'hour'))

st.subheader(f"# of Incidents by Month and Day")
st.plotly_chart(get_heatmap(fire_inc, 'month', 'day'))


##display map of incident clusters
st.subheader(f"Incident Clusters")
st.plotly_chart(get_cluster_map(fire_inc, dfd_locs))

##display prediction based on user inputs
inj_prob = make_prediction(in_date, in_time, lat, lon, light_coords, dfd_coords)
inj_prob = str(inj_prob)

# ## Use the sidebar on the left to make a prediction on the likelihood of
# injury or fatality. 

st.subheader(f"Likelihood of Injury or Fatality: {inj_prob}")


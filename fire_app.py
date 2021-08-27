from sklearn import cluster
import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.express as px
import plotly
import plotly.graph_objects as go
from scipy.spatial import distance
import xgboost as xgb
from joblib import dump, load

@st.cache
def load_data():
    """Loads relevant data from csv files

    Returns:
        [type]: [description]
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
        in_date ([type]): [description]
        in_time ([type]): [description]
        in_lat ([type]): [description]
        in_lon ([type]): [description]
        green_lights ([type]): [description]
        dfd_locs ([type]): [description]

    Returns:
        [type]: [description]
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
    cluster
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

    

    model = load("boost.joblib")
    model
    st.write(model.get_booster())

    model_features = model.get_booster().feature_names

    ##create indicator vars for cluster columns
    cluster_cols = [x for x in model_features if "cluster" in x]
    pred_df[cluster_cols] = 0
    cluster_col = "cluster_" + str(cluster)
    pred_df[cluster_col] = 1

    pred_df = pred_df[model_features] ##reorder columns

    prediction = model.predict_proba(pred_df)[0,1]

    return round(prediction, 2)

@st.cache
def get_cluster(lat, lon):
    clustering = load('cluster_model.joblib')   # long = x, lat = y
    return clustering.predict([[lat, lon]])[0]

def get_distances(lat, lon, light_coords, dfd_coords):
    stn_dist = distance.cdist(np.array([(lon, lat)]), dfd_coords).min(axis=1)[0]
    light_dist = distance.cdist(np.array([(lon, lat)]), light_coords).min(axis=1)[0]
    
    return stn_dist, light_dist

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

@st.cache()
def get_heatmap(fire_inc):
    dow_hour_counts = fire_inc.groupby(by=['DoW', 'hour'])['injury_or_fatality'].count().reset_index()
    dow_hour_counts = dow_hour_counts.pivot(index='DoW', columns='hour', values='injury_or_fatality')
    return px.imshow(dow_hour_counts)


st.title('Analysis of Fire Incidents in City of Detroit')
st.sidebar.title("Enter Prediction Inputs")
in_date = st.sidebar.date_input("Date")
in_time = st.sidebar.time_input("Time", )
lat = 42.33
lon = -83.05
lat = st.sidebar.number_input("Latitude", min_value=42.3, max_value=42.45, value=lat)
lon = st.sidebar.number_input("Longitude", min_value=-83.20, max_value=-82.95, value=lon)
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
dfd_coords = np.array(list(zip(dfd_locs['X'], dfd_locs['Y'])))
light_coords = np.array(list(zip(dfd_locs['X'], dfd_locs['Y'])))

st.subheader(f"# of Incidents by Day of Week and Hour")
st.plotly_chart(get_heatmap(fire_inc))

inj_prob = make_prediction(in_date, in_time, lat, lon, light_coords, dfd_coords)
inj_prob = str(inj_prob)

st.subheader(f"Likelihood of Injury or Fatality: {inj_prob}")


import os
import streamlit as st
import pandas as pd
import hopsworks
from datetime import date, timedelta
import matplotlib.pyplot as plt


FG_NAME = "police_events"
FG_VERSION = 2

PRED_FG_NAME = "police_predictions"
PRED_FG_VERSION = 6


# -----------------------------
# Config
# -----------------------------
try:
    import config
    API_KEY = config.HOPSWORKS_API_KEY
    PROJECT_NAME = config.HOPSWORKS_PROJECT_NAME
except Exception:
    API_KEY = os.environ.get("HOPSWORKS_API_KEY")
    PROJECT_NAME = os.environ.get("HOPSWORKS_PROJECT_NAME", "id2223_lab1_G22")

if not API_KEY:
    st.error("Missing HOPSWORKS_API_KEY (set it in env vars or src/config.py).")
    st.stop()

TITLE = "Incident and Crime Forecast"
DESCRIPTION = \
    "Location-based incident and crime forecast based on events reported by the Swedish police."\
    " Forecasts are based on temporal and weatherly features"\
    " and predict the proportion of events the next day."\
    "\n\n"\
    "Current events can be found here:\n\n"\
    "https://polisen.se/aktuellt/polisens-nyheter/"
st.set_page_config(page_title=TITLE, page_icon="👮", layout="centered")
st.title(TITLE)
st.write(DESCRIPTION)


@st.cache_data(show_spinner=False, ttl=600)
def load_predictions():
    """
    Read unique cities from the feature group.
    Cached for 10 minutes to reduce Hopsworks reads.
    """
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(PRED_FG_NAME, version=PRED_FG_VERSION)
    df = fg.read()
    return df


@st.cache_data(show_spinner=False, ttl=600)
def load_events():
    project = hopsworks.login(api_key_value=API_KEY, project=PROJECT_NAME)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(FG_NAME, version=FG_VERSION)
    df = fg.read()
    return df


tomorrow = pd.to_datetime(date.today() + timedelta(days=1)).date()
yesterday = pd.to_datetime(date.today() - timedelta(days=1)).date()

pred_df = load_predictions()
pred_df = pred_df.rename(
    columns={
        "predicted_type_group": "Type",
    }
)
tomorrow_pred_df = pred_df[pred_df["date"].dt.date == tomorrow]
#yest_pred_df = pred_df[pred_df["date"].dt.date == yesterday]

events_df = load_events()
events_df = events_df.rename(
    columns={
        "type_group": "Type",
    }
)
events_df = events_df[events_df["datetime"].dt.date == yesterday]


locations = sorted(tomorrow_pred_df["city"].dropna().unique().tolist())
if not locations:
    st.warning(f"No cities found in Feature Group {PRED_FG_NAME}.")
    st.stop()

# -----------------------------
# Sidebar
# -----------------------------
DEFAULT_CITY = "Stockholm"
st.sidebar.header("Menu")
selected_location = st.sidebar.selectbox("Location", locations, index=locations.index(DEFAULT_CITY))

selected_loc_preds_df = tomorrow_pred_df[tomorrow_pred_df["city"] == selected_location]
selected_loc_counts = selected_loc_preds_df["Type"].value_counts().sort_index()
selected_loc_dist = selected_loc_counts / len(selected_loc_preds_df)
selected_loc_dist = selected_loc_dist.rename("Proportion")

total_pred_counts = tomorrow_pred_df["Type"].value_counts().sort_index()
total_pred_dist = total_pred_counts / len(tomorrow_pred_df)
total_pred_dist = total_pred_dist.rename("Proportion")

total_past_counts = events_df["Type"].value_counts().sort_index()
total_past_dist = total_past_counts / len(events_df)
total_past_dist = total_past_dist.rename("Proportion")

#yest_pred_counts = yest_pred_df["Type"].value_counts().sort_index()
#yest_pred_dist = yest_pred_counts / len(yest_pred_df)
#yest_pred_dist = yest_pred_dist.rename("Proportion")

all_types = sorted(set(total_past_dist.index) | set(total_pred_dist.index))
total_past_dist = total_past_dist.reindex(all_types, fill_value=0)
#yest_pred_dist = yest_pred_dist.reindex(all_types, fill_value=0)


# -----------------------------
# Layout
# -----------------------------
st.subheader(f"{selected_location}, Tomorrow")
st.bar_chart(selected_loc_dist)

st.divider()
st.subheader(f"Sweden, Tomorrow")
st.bar_chart(total_pred_dist)
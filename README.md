# Incident and Crime Forecast

Created by Elmira Mirzaee and Johannes Arenander, 2026


## About

This project was written as part of the course in Scalable Machine Learning and Deep Learning, at KTH Royal Institute of Technology during the winter of 2025/2026. In this project, we built a location-based incident and crime forecasting model based on events reported by the Swedish Police. The forecasts are based on temporal and weatherly features and predict the proportion of events the next day.


## Pipeline

The following three parts make up the project pipeline:

1. A daily feature pipeline that retrieves new events from Polisen [https://polisen.se/api/events] and weatherly features from Open-Meteo [https://open-meteo.com/] and uploads them to a Hopsworks feature store [https://www.hopsworks.ai/].

2. A weekly training pipeline that trains an XGBoost model to predict the most likely events based on the features and uploads it to a model registry in Hopsworks.

3. A daily inference pipeline that fetches weatherly forecasts from Open-Meteo, makes predictions for the next day and uploads them to the feature store.


## Issues

The API retrieves the last 500 events, and it is not possible to fetch historical events retroactively. The model does not perform as well as it could at the creation of the project, but it is expected to improve with time as more data is collected.


## UI

The UI is a simple dashboard built with Streamlit [https://streamlit.io/].

View it here:

https://huggingface.co/spaces/johnaren/police-project-id2223

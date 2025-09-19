import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------
# Sample Synthetic Data (replace with your CSV)
# ------------------------
np.random.seed(42)
rows = 300

data = pd.DataFrame({
    'Date': pd.date_range(start="2025-01-01", periods=rows, freq='D'),
    'Weather': np.random.choice(['Clear', 'Rain', 'Storm', 'Fog'], rows),
    'Traffic': np.random.choice(['Low', 'Medium', 'High'], rows),
    'VehicleType': np.random.choice(['Truck', 'Van', 'Bike'], rows),
    'Area': np.random.choice(['Urban', 'Suburban', 'Rural'], rows),
    'Category': np.random.choice(['Food', 'Electronics', 'Clothing'], rows),
    'DeliveryTime': np.random.normal(30, 10, rows).round(1)
})

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="LogiSight Real-Time Dashboard", layout="wide")
st.title("📊 LogiSight Interactive Delivery Trends Dashboard")

st.sidebar.header("🔍 Apply Filters")

# Multi-select filters
weather_filter = st.sidebar.multiselect("Weather", options=data['Weather'].unique(), default=[])
traffic_filter = st.sidebar.multiselect("Traffic", options=data['Traffic'].unique(), default=[])
vehicle_filter = st.sidebar.multiselect("Vehicle Type", options=data['VehicleType'].unique(), default=[])
area_filter = st.sidebar.multiselect("Area", options=data['Area'].unique(), default=[])
category_filter = st.sidebar.multiselect("Product Category", options=data['Category'].unique(), default=[])

# ------------------------
# Filtering logic
# ------------------------
filtered_data = data.copy()

if weather_filter:
    filtered_data = filtered_data[filtered_data['Weather'].isin(weather_filter)]
if traffic_filter:
    filtered_data = filtered_data[filtered_data['Traffic'].isin(traffic_filter)]
if vehicle_filter:
    filtered_data = filtered_data[filtered_data['VehicleType'].isin(vehicle_filter)]
if area_filter:
    filtered_data = filtered_data[filtered_data['Area'].isin(area_filter)]
if category_filter:
    filtered_data = filtered_data[filtered_data['Category'].isin(category_filter)]

# ------------------------
# Show dynamic graphs
# ------------------------
if filtered_data.empty:
    st.warning("⚠️ No data available for the selected filters. Try adjusting your selections.")
else:
    st.subheader("📈 Real-Time Graphs Based on Selection")

    # Base graph - delivery time trend
    fig = px.line(filtered_data, x="Date", y="DeliveryTime", color="Weather",
                  title="Delivery Time Trend (Filtered)", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    # Generate separate graphs for each selected filter combination
    st.subheader("🔎 Breakdown by Selected Conditions")

    grouping_columns = []
    if weather_filter: grouping_columns.append("Weather")
    if traffic_filter: grouping_columns.append("Traffic")
    if vehicle_filter: grouping_columns.append("VehicleType")
    if area_filter: grouping_columns.append("Area")
    if category_filter: grouping_columns.append("Category")

    if grouping_columns:
        grouped = filtered_data.groupby(grouping_columns + ["Date"], as_index=False).agg({"DeliveryTime":"mean"})

        for group_vals, df_subset in grouped.groupby(grouping_columns):
            group_title = ", ".join([f"{col}: {val}" for col, val in zip(grouping_columns, group_vals if isinstance(group_vals, tuple) else [group_vals])])

            fig = px.line(df_subset, x="Date", y="DeliveryTime",
                          title=f"📊 Trend for {group_title}", markers=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("ℹ️ Select one or more filters to see condition-specific graphs.")

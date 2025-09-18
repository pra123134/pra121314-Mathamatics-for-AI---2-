import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", palette="muted", font_scale=1.1)

st.set_page_config(page_title="Delivery Dashboard", layout="wide")

st.title("Interactive Delivery Dashboard")
uploaded_file = st.file_uploader("Upload your delivery CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    df.dropna(subset=['DeliveryID', 'DeliveryTime', 'VehicleType'], inplace=True)
    
    categorical_cols = ['Weather', 'Traffic', 'VehicleType', 'Area', 'Category']
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Age groups for scatter plot
    if 'Agent_Age' in df.columns:
        df['AgeGroup'] = pd.cut(
            df['Agent_Age'],
            bins=[0, 25, 40, 100],
            labels=['<25', '25–40', '40+']
        )
    
    # Late delivery flag
    avg_time = df["DeliveryTime"].mean()
    threshold = avg_time + df["DeliveryTime"].std()
    df["LateDeliveryFlag"] = np.where(df["DeliveryTime"] > threshold, 1, 0)
  
    st.sidebar.header("Filters")
    
    weather_filter = st.sidebar.multiselect(
        "Weather", options=df["Weather"].unique(), default=df["Weather"].unique()
    )
    traffic_filter = st.sidebar.multiselect(
        "Traffic", options=df["Traffic"].unique(), default=df["Traffic"].unique()
    )
    vehicle_filter = st.sidebar.multiselect(
        "Vehicle Type", options=df["VehicleType"].unique(), default=df["VehicleType"].unique()
    )
    area_filter = st.sidebar.multiselect(
        "Area", options=df["Area"].unique(), default=df["Area"].unique()
    )
    category_filter = st.sidebar.multiselect(
        "Product Category", options=df["Category"].unique(), default=df["Category"].unique()
    )
    
    # Apply filters
    filtered_df = df[
        (df["Weather"].isin(weather_filter)) &
        (df["Traffic"].isin(traffic_filter)) &
        (df["VehicleType"].isin(vehicle_filter)) &
        (df["Area"].isin(area_filter)) &
        (df["Category"].isin(category_filter))
    ]
    
    avg_delivery = filtered_df["DeliveryTime"].mean()
    late_percentage = filtered_df["LateDeliveryFlag"].mean() * 100
    
    col1, col2 = st.columns(2)
    col1.metric("Average Delivery Time (mins)", f"{avg_delivery:.2f}")
    col2.metric("Late Deliveries (%)", f"{late_percentage:.2f}%")
  
    st.subheader("Delay Analyzer: Avg Delivery Time by Weather & Traffic")
    plt.figure(figsize=(10,5))
    sns.barplot(data=filtered_df, x="Weather", y="DeliveryTime", hue="Traffic", ci="sd")
    plt.ylabel("Avg Delivery Time (mins)")
    plt.xlabel("Weather")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("Vehicle Comparison: Avg Delivery Time by Vehicle")
    plt.figure(figsize=(8,5))
    sns.barplot(data=filtered_df, x="VehicleType", y="DeliveryTime", ci="sd")
    plt.ylabel("Avg Delivery Time (mins)")
    plt.xlabel("Vehicle Type")
    st.pyplot(plt.gcf())
    plt.clf()
    
    if 'Agent_Rating' in filtered_df.columns and 'Agent_Age' in filtered_df.columns:
        st.subheader("Agent Performance: Rating vs Delivery Time (by Age Group)")

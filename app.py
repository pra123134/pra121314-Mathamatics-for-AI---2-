import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Delivery Dashboard", layout="wide")
st.title("🚚 Delivery Performance Dashboard")

st.sidebar.header("Upload Delivery Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
   
    df.dropna(subset=['DeliveryID', 'DeliveryTime', 'VehicleType'], inplace=True)
    
    categorical_cols = ['Weather', 'Traffic', 'VehicleType', 'Area', 'Category']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Age Grouping
    if 'Agent_Age' in df.columns:
        df['AgeGroup'] = pd.cut(
            df['Agent_Age'], bins=[0,25,40,100], labels=['<25','25–40','40+']
        )
    
    # Late Delivery Flag
    avg_time = df['DeliveryTime'].mean()
    threshold = avg_time + df['DeliveryTime'].std()
    df['LateDeliveryFlag'] = np.where(df['DeliveryTime'] > threshold, 1, 0)
   
    st.sidebar.header("Filters")
    
    # Multi-select filters
    filter_weather = st.sidebar.multiselect(
        "Weather", options=df['Weather'].unique(), default=df['Weather'].unique()
    )
    filter_traffic = st.sidebar.multiselect(
        "Traffic", options=df['Traffic'].unique(), default=df['Traffic'].unique()
    )
    filter_vehicle = st.sidebar.multiselect(
        "Vehicle Type", options=df['VehicleType'].unique(), default=df['VehicleType'].unique()
    )
    filter_area = st.sidebar.multiselect(
        "Area", options=df['Area'].unique(), default=df['Area'].unique()
    )
    filter_category = st.sidebar.multiselect(
        "Product Category", options=df['Category'].unique(), default=df['Category'].unique()
    )
    
    # Apply filters
    filtered_df = df[
        (df['Weather'].isin(filter_weather)) &
        (df['Traffic'].isin(filter_traffic)) &
        (df['VehicleType'].isin(filter_vehicle)) &
        (df['Area'].isin(filter_area)) &
        (df['Category'].isin(filter_category))
    ]
   
    col1, col2 = st.columns(2)
    
    with col1:
        avg_delivery = filtered_df['DeliveryTime'].mean()
        st.metric("Average Delivery Time (mins)", f"{avg_delivery:.2f}")
    
    with col2:
        late_percentage = filtered_df['LateDeliveryFlag'].mean() * 100
        st.metric("Late Deliveries (%)", f"{late_percentage:.2f}%")
   
    st.subheader("📊 Delay Analyzer: Avg Delivery Time by Weather & Traffic")
    plt.figure(figsize=(10,5))
    sns.barplot(
        data=filtered_df,
        x="Weather",
        y="DeliveryTime",
        hue="Traffic",
        ci="sd"
    )
    plt.ylabel("Average Delivery Time (mins)")
    plt.xlabel("Weather")
    plt.legend(title="Traffic")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("🚗 Vehicle Comparison: Avg Delivery Time by Vehicle")
    plt.figure(figsize=(8,5))
    sns.barplot(data=filtered_df, x="VehicleType", y="DeliveryTime", ci="sd")
    plt.ylabel("Average Delivery Time (mins)")
    plt.xlabel("Vehicle Type")
    st.pyplot(plt.gcf())
    plt.clf()
    
    if 'Agent_Rating' in df.columns and 'Agent_Age' in df.columns:
        st.subheader("👨‍💼 Agent Performance: Rating vs Delivery Time")
        plt.figure(figsize=(10,5))
        sns.scatterplot(
            data=filtered_df,
            x="Agent_Rating",
            y="DeliveryTime",
            hue="AgeGroup",
            palette="deep"
        )
        plt.ylabel("Delivery Time (mins)")
        plt.xlabel("Agent Rating")
        plt.legend(title="Age Group")
        st.pyplot(plt.gcf())
        plt.clf()
    
    st.subheader("📍 Area Heatmap: Avg Delivery Time by Area")
    area_summary = filtered_df.groupby("Area")["DeliveryTime"].mean().reset_index()
    plt.figure(figsize=(10,5))
    sns.heatmap(area_summary.set_index('Area').T, annot=True, cmap="YlOrRd")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("📦 Category Visualizer: Delivery Time Distribution by Product Category")
    plt.figure(figsize=(10,5))
    sns.boxplot(data=filtered_df, x="Category", y="DeliveryTime")
    plt.ylabel("Delivery Time (mins)")
    plt.xlabel("Category")
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Optional Visuals
    if 'DeliveryDate' in df.columns:
        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], errors='coerce')
        filtered_df['Month'] = filtered_df['DeliveryDate'].dt.to_period('M')
        st.subheader("📈 Monthly Trend: Avg Delivery Time")
        monthly_summary = filtered_df.groupby('Month')['DeliveryTime'].mean().reset_index()
        plt.figure(figsize=(10,4))
        sns.lineplot(data=monthly_summary, x='Month', y='DeliveryTime', marker='o')
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.clf()
    
    st.subheader("⏱ Delivery Time Distribution")
    plt.figure(figsize=(8,4))
    sns.histplot(filtered_df['DeliveryTime'], kde=True, bins=30)
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("% Late Deliveries by Traffic")
    traffic_summary = filtered_df.groupby('Traffic')['LateDeliveryFlag'].mean().reset_index()
    traffic_summary['LateDeliveryFlag'] *= 100
    plt.figure(figsize=(8,4))
    sns.barplot(data=traffic_summary, x='Traffic', y='LateDeliveryFlag')
    plt.ylabel("Late Deliveries (%)")
    st.pyplot(plt.gcf())
    plt.clf()
    
    st.subheader("Agent Count per Area")
    plt.figure(figsize=(10,4))
    sns.countplot(data=filtered_df, x='Area', order=filtered_df['Area'].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()

else:
    st.info("Please upload a CSV file to visualize the delivery data.")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Last Mile Delivery Dashboard", layout="wide")
st.title("🚚 Last Mile Delivery Dashboard")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
   
    df.dropna(subset=['Order_ID', 'Delivery_Time', 'Vehicle'], inplace=True)
    
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Age groups for scatter plot
    if 'Agent_Age' in df.columns:
        df['AgentAgeGroup'] = pd.cut(
            df['Agent_Age'],
            bins=[0,25,40,100],
            labels=['<25','25–40','40+']
        )
    
    # Late delivery flag
    avg_time = df['Delivery_Time'].mean()
    threshold = avg_time + df['Delivery_Time'].std()
    df['LateDeliveryFlag'] = np.where(df['Delivery_Time'] > threshold, 1, 0)
   
    st.sidebar.header("Filters")
    filter_weather = st.sidebar.multiselect("Weather", options=df['Weather'].unique(), default=df['Weather'].unique())
    filter_traffic = st.sidebar.multiselect("Traffic", options=df['Traffic'].unique(), default=df['Traffic'].unique())
    filter_vehicle = st.sidebar.multiselect("Vehicle Type", options=df['Vehicle'].unique(), default=df['Vehicle'].unique())
    filter_area = st.sidebar.multiselect("Area", options=df['Area'].unique(), default=df['Area'].unique())
    filter_category = st.sidebar.multiselect("Category", options=df['Category'].unique(), default=df['Category'].unique())
    
    filtered_df = df[
        (df['Weather'].isin(filter_weather)) &
        (df['Traffic'].isin(filter_traffic)) &
        (df['Vehicle'].isin(filter_vehicle)) &
        (df['Area'].isin(filter_area)) &
        (df['Category'].isin(filter_category))
    ]
    
    col1, col2 = st.columns(2)
    col1.metric("Average Delivery Time (mins)", f"{filtered_df['Delivery_Time'].mean():.2f}")
    col2.metric("Late Deliveries (%)", f"{filtered_df['LateDeliveryFlag'].mean()*100:.2f}%")
    
    sns.set(style="whitegrid", palette="muted", font_scale=1.0)
    
    # -------- 1. Delay Analyzer --------
    st.subheader("Delay Analyzer: Avg Delivery Time by Weather & Traffic")
    plt.figure(figsize=(10,5))
    sns.barplot(data=filtered_df, x="Weather", y="Delivery_Time", hue="Traffic", ci="sd")
    plt.ylabel("Avg Delivery Time (mins)")
    plt.xlabel("Weather")
    plt.legend(title="Traffic")
    st.pyplot(plt.gcf())
    plt.clf()
    
    # -------- 2. Vehicle Comparison --------
    st.subheader("Vehicle Comparison: Avg Delivery Time by Vehicle")
    plt.figure(figsize=(8,5))
    sns.barplot(data=filtered_df, x="Vehicle", y="Delivery_Time", ci="sd")
    plt.ylabel("Avg Delivery Time (mins)")
    plt.xlabel("Vehicle Type")
    st.pyplot(plt.gcf())
    plt.clf()
    
    # -------- 3. Agent Performance Scatter --------
    if 'Agent_Rating' in df.columns and 'Agent_Age' in df.columns:
        st.subheader("Agent Performance: Rating vs Delivery Time")
        plt.figure(figsize=(10,5))
        sns.scatterplot(data=filtered_df, x="Agent_Rating", y="Delivery_Time", hue="AgentAgeGroup", palette="deep")
        plt.ylabel("Delivery Time (mins)")
        plt.xlabel("Agent Rating")
        plt.legend(title="Age Group")
        st.pyplot(plt.gcf())
        plt.clf()
    
    # -------- 4. Area Heatmap --------
    st.subheader("Area Heatmap: Avg Delivery Time by Area")
    area_summary = filtered_df.groupby("Area")["Delivery_Time"].mean().reset_index()
    plt.figure(figsize=(10,5))
    sns.heatmap(area_summary.set_index('Area').T, annot=True, cmap="YlOrRd")
    st.pyplot(plt.gcf())
    plt.clf()
    
    # -------- 5. Category Visualizer --------
    st.subheader("Category Visualizer: Delivery Time Distribution by Product Category")
    plt.figure(figsize=(10,5))
    sns.boxplot(data=filtered_df, x="Category", y="Delivery_Time")
    plt.ylabel("Delivery Time (mins)")
    plt.xlabel("Category")
    st.pyplot(plt.gcf())
    plt.clf()
    
    # -------- Optional: Monthly Trend --------
    if "DeliveryDate" in df.columns:
        df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors="coerce")
        filtered_df["Month"] = filtered_df["DeliveryDate"].dt.to_period("M")
        st.subheader("Monthly Trend: Avg Delivery Time")
        monthly_summary = filtered_df.groupby("Month")["Delivery_Time"].mean().reset_index()
        plt.figure(figsize=(10,4))
        sns.lineplot(data=monthly_summary, x="Month", y="Delivery_Time", marker="o")
        plt.xticks(rotation=45)
        st.pyplot(plt.gcf())
        plt.clf()
    
    # Delivery Time Distribution
    st.subheader("Delivery Time Distribution")
    plt.figure(figsize=(8,4))
    sns.histplot(filtered_df['Delivery_Time'], kde=True, bins=30)
    st.pyplot(plt.gcf())
    plt.clf()
    
    # % Late Deliveries by Traffic
    st.subheader("% Late Deliveries by Traffic")
    traffic_summary = filtered_df.groupby('Traffic')['LateDeliveryFlag'].mean().reset_index()
    traffic_summary['LateDeliveryFlag'] *= 100
    plt.figure(figsize=(8,4))
    sns.barplot(data=traffic_summary, x='Traffic', y='LateDeliveryFlag')
    plt.ylabel("Late Deliveries (%)")
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Agent Count per Area
    st.subheader("Agent Count per Area")
    plt.figure(figsize=(10,4))
    sns.countplot(data=filtered_df, x='Area', order=filtered_df['Area'].value_counts().index)
    plt.xticks(rotation=45)
    st.pyplot(plt.gcf())
    plt.clf()
    
else:
    st.info("Please upload a CSV file to visualize the delivery data.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="🚚 Last Mile Delivery Dashboard", layout="wide")
st.title("🚚 Interactive Delivery Performance Dashboard")
st.markdown("Explore delivery trends, identify bottlenecks, and forecast potential delays dynamically.")

# ---------------- CSV Upload ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    # ---------------- Data Cleaning ----------------
    df.dropna(subset=['Order_ID', 'Delivery_Time', 'Vehicle'], inplace=True)
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    if 'Agent_Age' in df.columns:
        df['AgentAgeGroup'] = pd.cut(df['Agent_Age'], bins=[0,25,40,100], labels=['<25','25–40','40+'])
    
    df['LateDeliveryFlag'] = np.where(df['Delivery_Time'] > df['Delivery_Time'].mean() + df['Delivery_Time'].std(), 1, 0)
    
    # Convert date
    if 'DeliveryDate' in df.columns:
        df['DeliveryDate'] = pd.to_datetime(df['DeliveryDate'], errors='coerce')
        df['Month'] = df['DeliveryDate'].dt.to_period('M')
    
    # ---------------- Sidebar Filters ----------------
    st.sidebar.header("Filters & Metrics")
    metric = st.sidebar.selectbox(
        "Select Metric for Graphs",
        ['Delivery_Time', 'LateDeliveryFlag', 'Agent_Rating']
    )
    x_axis = st.sidebar.selectbox(
        "Select X-axis",
        ['Weather','Traffic','Vehicle','Area','Category','Month','AgentAgeGroup']
    )
    color_group = st.sidebar.selectbox(
        "Color by",
        ['Weather','Traffic','Vehicle','Area','Category','AgentAgeGroup','None']
    )
    
    filter_weather = st.sidebar.multiselect("Filter Weather", options=df['Weather'].unique(), default=df['Weather'].unique())
    filter_traffic = st.sidebar.multiselect("Filter Traffic", options=df['Traffic'].unique(), default=df['Traffic'].unique())
    filter_vehicle = st.sidebar.multiselect("Filter Vehicle", options=df['Vehicle'].unique(), default=df['Vehicle'].unique())
    filter_area = st.sidebar.multiselect("Filter Area", options=df['Area'].unique(), default=df['Area'].unique())
    filter_category = st.sidebar.multiselect("Filter Category", options=df['Category'].unique(), default=df['Category'].unique())
    
    df_filtered = df[
        (df['Weather'].isin(filter_weather)) &
        (df['Traffic'].isin(filter_traffic)) &
        (df['Vehicle'].isin(filter_vehicle)) &
        (df['Area'].isin(filter_area)) &
        (df['Category'].isin(filter_category))
    ]
    
    # ---------------- Metrics ----------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Delivery Time", f"{df_filtered['Delivery_Time'].mean():.2f} mins")
    col2.metric("Late Deliveries (%)", f"{df_filtered['LateDeliveryFlag'].mean()*100:.2f}%")
    col3.metric("Total Deliveries", f"{len(df_filtered)}")
    col4.metric("Top Vehicle", df_filtered.groupby('Vehicle')['Delivery_Time'].mean().idxmin())
    
    # ---------------- Dynamic Plot ----------------
    st.subheader(f"Interactive Trend: {metric} vs {x_axis}")
    if color_group != "None":
        fig = px.bar(df_filtered, x=x_axis, y=metric, color=color_group,
                     hover_data=['Delivery_Time','Agent_Rating'], barmode='group')
    else:
        fig = px.bar(df_filtered, x=x_axis, y=metric, hover_data=['Delivery_Time','Agent_Rating'])
    st.plotly_chart(fig, use_container_width=True)
    
    # ---------------- Forecast / Future Trend Placeholder ----------------
    st.subheader("📈 Future Trend Preview (Moving Average Forecast)")
    if 'DeliveryDate' in df_filtered.columns:
        df_time = df_filtered.groupby('DeliveryDate')['Delivery_Time'].mean().reset_index()
        df_time['MovingAvg'] = df_time['Delivery_Time'].rolling(window=7, min_periods=1).mean()
        fig2 = px.line(df_time, x='DeliveryDate', y='Delivery_Time', title="Daily Avg Delivery Time", markers=True)
        fig2.add_scatter(x=df_time['DeliveryDate'], y=df_time['MovingAvg'], mode='lines', name='7-Day Moving Avg')
        st.plotly_chart(fig2, use_container_width=True)
    
    st.success("Interactive delivery dashboard ready! Adjust dropdowns and filters to explore trends dynamically.")
else:
    st.info("Please upload a CSV file to start exploring delivery trends.")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time

st.set_page_config(page_title="🚀 Live Delivery Dashboard", layout="wide")
st.title("📈 Real-Time Last Mile Delivery Dashboard")
st.markdown("Simulated real-time delivery metrics, trends, and operational insights.")

# ---------------- CSV Upload ----------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df_original = pd.read_csv(uploaded_file)
    
    # ---------------- Data Cleaning ----------------
    df_original.dropna(subset=['Order_ID', 'Delivery_Time', 'Vehicle'], inplace=True)
    categorical_cols = ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category']
    df_original[categorical_cols] = df_original[categorical_cols].fillna("Unknown")
    
    numeric_cols = df_original.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_original[col] = df_original[col].fillna(df_original[col].median())
    
    if 'Agent_Age' in df_original.columns:
        df_original['AgentAgeGroup'] = pd.cut(df_original['Agent_Age'], bins=[0,25,40,100], labels=['<25','25–40','40+'])
    
    df_original['LateDeliveryFlag'] = np.where(df_original['Delivery_Time'] > df_original['Delivery_Time'].mean() + df_original['Delivery_Time'].std(), 1, 0)
    
    # Convert date
    if 'DeliveryDate' in df_original.columns:
        df_original['DeliveryDate'] = pd.to_datetime(df_original['DeliveryDate'], errors='coerce')
        df_original['Month'] = df_original['DeliveryDate'].dt.to_period('M')
    
    # ---------------- Sidebar Filters ----------------
    st.sidebar.header("Filters & Options")
    metric = st.sidebar.selectbox("Select Metric", ['Delivery_Time', 'LateDeliveryFlag', 'Agent_Rating'])
    x_axis = st.sidebar.selectbox("Select X-axis", ['Weather','Traffic','Vehicle','Area','Category','Month','AgentAgeGroup'])
    color_group = st.sidebar.selectbox("Color by", ['Weather','Traffic','Vehicle','Area','Category','AgentAgeGroup','None'])
    
    filter_weather = st.sidebar.multiselect("Filter Weather", df_original['Weather'].unique(), default=df_original['Weather'].unique())
    filter_traffic = st.sidebar.multiselect("Filter Traffic", df_original['Traffic'].unique(), default=df_original['Traffic'].unique())
    filter_vehicle = st.sidebar.multiselect("Filter Vehicle", df_original['Vehicle'].unique(), default=df_original['Vehicle'].unique())
    filter_area = st.sidebar.multiselect("Filter Area", df_original['Area'].unique(), default=df_original['Area'].unique())
    filter_category = st.sidebar.multiselect("Filter Category", df_original['Category'].unique(), default=df_original['Category'].unique())
    
    # Containers for live updates
    metrics_container = st.container()
    chart_container = st.contain_

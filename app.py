import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Last Mile Delivery Dashboard", layout="wide")
st.title("🚚 Last Mile Delivery Dashboard")
st.markdown("Interactively explore delivery trends, identify delays, and uncover operational insights.")

# -------------------------
# CSV Upload
# -------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # -------------------------
    # Data Cleaning & Preparation
    # -------------------------
    required_cols = ['Order_ID', 'Delivery_Time', 'Vehicle']
    for col in required_cols:
        if col not in df.columns:
            st.error(f"❌ Missing required column: {col}")
            st.stop()

    df.dropna(subset=required_cols, inplace=True)

    categorical_cols = [col for col in ['Weather', 'Traffic', 'Vehicle', 'Area', 'Category'] if col in df.columns]
    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # Make age groups if available
    if 'Agent_Age' in df.columns:
        df['AgentAgeGroup'] = pd.cut(df['Agent_Age'], bins=[0, 25, 40, 100], labels=['<25', '25–40', '40+'])

    # Late delivery flag
    avg_time = df['Delivery_Time'].mean()
    threshold = avg_time + df['Delivery_Time'].std()
    df['LateDeliveryFlag'] = np.where(df['Delivery_Time'] > threshold, 1, 0)

    # -------------------------
    # Sidebar Filters
    # -------------------------
    st.sidebar.header("Filter Deliveries")

    def sidebar_filter(colname):
        if colname in df.columns:
            return st.sidebar.multiselect(colname, options=df[colname].unique(), default=df[colname].unique())
        return None

    filter_weather = sidebar_filter("Weather")
    filter_traffic = sidebar_filter("Traffic")
    filter_vehicle = sidebar_filter("Vehicle")
    filter_area = sidebar_filter("Area")
    filter_category = sidebar_filter("Category")

    # Apply filters safely
    filtered_df = df.copy()

    if filter_weather is not None:
        filtered_df = filtered_df[filtered_df["Weather"].isin(filter_weather)]
    if filter_traffic is not None:
        filtered_df = filtered_df[filtered_df["Traffic"].isin(filter_traffic)]
    if filter_vehicle is not None:
        filtered_df = filtered_df[filtered_df["Vehicle"].isin(filter_vehicle)]
    if filter_area is not None:
        filtered_df = filtered_df[filtered_df["Area"].isin(filter_area)]
    if filter_category is not None:
        filtered_df = filtered_df[filtered_df["Category"].isin(filter_category)]

    # -------------------------
    # Key Metrics
    # -------------------------
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Delivery Time (mins)", f"{filtered_df['Delivery_Time'].mean():.2f}")
    col2.metric("Late Deliveries (%)", f"{filtered_df['LateDeliveryFlag'].mean()*100:.2f}%")
    col3.metric("Total Deliveries", len(filtered_df))

    # Download button
    st.download_button(
        label="📥 Download Filtered Data",
        data=filtered_df.to_csv(index=False),
        file_name='filtered_delivery_data.csv',
        mime='text/csv'
    )

    sns.set(style="whitegrid", palette="muted")

    # -------------------------
    # Tabs for Visuals
    # -------------------------
    tabs = st.tabs(["Compulsory Visuals", "Optional Visuals"])

    # -------- Compulsory Visuals --------
    with tabs[0]:

        if "Weather" in df.columns and "Traffic" in df.columns:
            st.subheader("Delay Analyzer: Avg Delivery Time by Weather & Traffic")
            plt.figure(figsize=(10, 5))
            sns.barplot(data=filtered_df, x="Weather", y="Delivery_Time", hue="Traffic", ci="sd")
            st.pyplot(plt.gcf())
            plt.clf()

        if "Vehicle" in df.columns:
            st.subheader("Vehicle Comparison")
            plt.figure(figsize=(10, 5))
            sns.barplot(data=filtered_df, x="Vehicle", y="Delivery_Time", ci="sd")
            st.pyplot(plt.gcf())
            plt.clf()

        if "Agent_Rating" in df.columns and "Agent_Age" in df.columns:
            st.subheader("Agent Performance: Rating vs Delivery Time")
            plt.figure(figsize=(10, 5))
            sns.scatterplot(
                data=filtered_df,
                x="Agent_Rating",
                y="Delivery_Time",
                hue="AgentAgeGroup"
            )
            st.pyplot(plt.gcf())
            plt.clf()

        if "Area" in df.columns:
            st.subheader("Area Heatmap")
            area_summary = filtered_df.groupby("Area")["Delivery_Time"].mean().reset_index()
            heatmap_df = area_summary.set_index('Area')
            plt.figure(figsize=(10, 4))
            sns.heatmap(heatmap_df.T, annot=True, cmap="YlOrRd")
            st.pyplot(plt.gcf())
            plt.clf()

        if "Category" in df.columns:
            st.subheader("Category Distribution")
            plt.figure(figsize=(10, 5))
            sns.boxplot(data=filtered_df, x="Category", y="Delivery_Time")
            st.pyplot(plt.gcf())
            plt.clf()

    # -------- Optional Visuals --------
    with tabs[1]:

        if "DeliveryDate" in df.columns:
            df["DeliveryDate"] = pd.to_datetime(df["DeliveryDate"], errors='coerce')
            filtered_df["Month"] = filtered_df["DeliveryDate"].dt.to_period("M")

            st.subheader("Monthly Trend")
            monthly_summary = filtered_df.groupby("Month")["Delivery_Time"].mean().reset_index()

            plt.figure(figsize=(10, 5))
            sns.lineplot(data=monthly_summary, x="Month", y="Delivery_Time", marker="o")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
            plt.clf()

        st.subheader("Delivery Time Distribution")
        plt.figure(figsize=(10, 4))
        sns.histplot(filtered_df["Delivery_Time"], bins=30, kde=True)
        st.pyplot(plt.gcf())
        plt.clf()

        if "Traffic" in df.columns:
            st.subheader("% Late Deliveries by Traffic")
            traffic_summary = filtered_df.groupby("Traffic")["LateDeliveryFlag"].mean() * 100
            plt.figure(figsize=(10, 4))
            sns.barplot(x=traffic_summary.index, y=traffic_summary.values)
            st.pyplot(plt.gcf())
            plt.clf()

        if "Area" in df.columns:
            st.subheader("Agent Count per Area")
            plt.figure(figsize=(10, 4))
            sns.countplot(data=filtered_df, x="Area", order=filtered_df["Area"].value_counts().index)
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
            plt.clf()

else:
    st.info("Please upload a CSV file to explore delivery trends.")

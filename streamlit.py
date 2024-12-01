import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os

# Set page configuration
st.set_page_config(
    page_title="Weather Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure a directory for saved plots exists
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Main function to run the Streamlit app
def main():
    # Add the logo and main title
    st.image("https://via.placeholder.com/150", width=120)  # Replace with your logo path
    st.title("Weather Prediction App 🌦")

    # Define tabs for navigation
    tabs = st.tabs([
        "Home Page",
        "Upload Data & EDA",
        "Correlation Study",
        "ARIMA Model",
        "Random Forest Model",
        "Prediction, Comparison & Graphs"
    ])

    # Home Page
    with tabs[0]:
        st.header("🏠 Home Page")
        st.write("""
        Welcome to the Weather Prediction App!  
        This tool leverages advanced statistical and machine learning models for weather prediction.

        *Features:*
        - Upload your dataset and explore it through interactive visualizations.
        - Analyze data relationships with correlation studies.
        - Train models like ARIMA and Random Forest.
        - Compare predictions and visualize results.
        """)

    # Upload Data & EDA
    with tabs[1]:
        st.header("📂 Upload Data & EDA")
        
        # Sidebar for file upload and EDA button
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])
        perform_eda_button = st.sidebar.button("Perform EDA")
        
        if uploaded_file is not None:
            try:
                data = load_data(uploaded_file)
                st.write("### Data Preview")
                st.write(data.head())
                
                # Perform EDA when the button is pressed
                if perform_eda_button:
                    perform_eda(data)
            except Exception as e:
                st.error(f"An error occurred while processing the file: {e}")
        else:
            st.warning("Please upload a CSV file to proceed.")

    # Correlation Study
    with tabs[2]:
        st.header("🔗 Correlation Study")
        st.write("Analyze relationships between variables using correlation coefficients:")
        st.write("- Pearson Correlation")
        st.write("- Heatmaps")
        st.write("🚧 Content under construction.")

    # ARIMA Model
    with tabs[3]:
        st.header("📈 ARIMA Model")
        
        if uploaded_file is not None:
            try:
                selected_col = st.selectbox("Select column for ARIMA analysis", data.columns)
                train_data = data[selected_col]
                st.write("Training ARIMA model...")
                model = SARIMAX(train_data, order=(3, 0, 1), seasonal_order=(1, 1, 0, 7))
                forecast_steps = st.number_input("Enter forecast steps:", min_value=1, max_value=100, value=10)
                forecast_series = sarima_forecast(model, forecast_steps)
                plot_forecast(train_data, forecast_series)
            except Exception as e:
                st.error(f"Error while running ARIMA: {e}")
        else:
            st.warning("Please upload a dataset to use this feature.")

    # Random Forest Model
    with tabs[4]:
        st.header("🌲 Random Forest Model")
        st.write("🚧 Content under construction.")

    # Prediction, Comparison & Graphs
    with tabs[5]:
        st.header("📊 Prediction, Comparison & Graphs")
        st.write("🚧 Content under construction.")

# Function to load the data
def load_data(file):
    try:
        data = pd.read_csv(file)
        return data
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")

# Function to perform EDA
def perform_eda(data):
    st.write("## Exploratory Data Analysis (EDA)")
    st.write("### Summary Statistics")
    st.write(data.describe())

    # Boxplots for numeric columns
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        st.write(f"### Boxplot for {col}")
        fig, ax = plt.subplots()
        ax.boxplot(data[col].dropna())
        ax.set_title(f"Boxplot for {col}")
        st.pyplot(fig)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(data[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns to generate a heatmap.")

    # Pairplot
    st.write("### Pairplot of Numeric Variables")
    if len(numeric_cols) > 1:
        fig = sns.pairplot(data[numeric_cols])
        st.pyplot(fig)
    else:
        st.write("Not enough numeric columns to generate a pairplot.")

    # Distribution plots
    st.write("### Distribution of Numeric Variables")
    for col in numeric_cols:
        st.write(f"#### Distribution for {col}")
        fig, ax = plt.subplots()
        sns.histplot(data[col].dropna(), kde=True, ax=ax, bins=30)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# Function to forecast using SARIMA
def sarima_forecast(model, steps):
    st.write("Forecasting future values...")
    results = model.fit(disp=False)
    forecast = results.forecast(steps=steps)
    return forecast

# Function to plot the forecast
def plot_forecast(original_series, forecast_series):
    fig, ax = plt.subplots()
    ax.plot(original_series, label='Original', color='blue')
    ax.plot(range(len(original_series), len(original_series) + len(forecast_series)),
            forecast_series, label='Forecast', color='red')
    ax.set_title("Original vs Forecast")
    plt.legend()
    st.pyplot(fig)

# Function to add sidebar
def add_sidebar(page_name):
    with st.sidebar:
        st.header(f"{page_name} Sidebar")
        st.write("Navigate between sections using the toolbar at the top.")
        st.write("- Upload and explore data.")
        st.write("- Train models and compare results.")

# Run the app
if _name_ == "_main_":
    main()

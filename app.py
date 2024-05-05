import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

# Load the trained model
def load_trained_model(model_path):
    return load_model(model_path)

def main():
    st.set_page_config(
        page_title="Landslide Detection App",
        page_icon=":mountain:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Set custom font and colors
    st.markdown(
        """
        <style>
        .stApp {
            font-family: 'Arial', sans-serif;
            color: #333333; /* Dark gray text */
            background-color: #f4f4f4; /* Light gray background */
        }
        .st-bq {
            background-color: #ffffff; /* White */
            padding: 1.5rem;
            border-radius: 0.5rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title(":mountain: Landslide Detection App :mountain:")
    st.sidebar.title(":file_folder: Upload New Data :file_folder:")

    # File uploader for new data
    uploaded_file = st.sidebar.file_uploader("Upload Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Load the new data for prediction
        new_data = pd.read_excel(uploaded_file)

        # Display the uploaded data with styled borders and shadows
        st.subheader(":clipboard: Uploaded Data :clipboard:")
        st.dataframe(new_data.style.set_properties(**{
            "border": "1px solid #dddddd",  # Light gray border
            "border-collapse": "collapse",
            "box-shadow": "0 4px 8px rgba(0, 0, 0, 0.1)",
            "padding": "0.5rem",
        }))

        # Add interactive checkbox to show/hide raw data
        show_raw_data = st.checkbox("Show Raw Data")
        if show_raw_data:
            st.write(new_data)

        # Preprocess the new data
        scaler = StandardScaler()
        X_new = scaler.fit_transform(new_data.values)

        # Load the trained model
        model = load_trained_model("landslide_detection_model.h5")

        # Make predictions
        predictions = model.predict(X_new)

        # Convert predictions to binary classes
        binary_predictions = (predictions > 0.5).astype(int)

        # Display predictions with styled borders and shadows
        st.subheader(":chart_with_upwards_trend: Predictions :chart_with_downwards_trend:")
        for i, prediction in enumerate(binary_predictions):
            if prediction == 1:
                st.error(f":warning: Data point {i+1}: Landslide is likely to occur :exclamation:")
            else:
                st.success(f":white_check_mark: Data point {i+1}: No landslide is likely to occur :sun_with_face:")

        # Create a bar chart to visualize the distribution of predictions
        st.subheader(":bar_chart: Prediction Distribution :bar_chart:")
        prediction_counts = pd.Series(binary_predictions.flatten()).value_counts()
        plt.figure(figsize=(8, 6))  # Adjust the size of the figure
        colors = ["#ffa07a", "#87ceeb"]  # Custom colors for the bars
        prediction_counts.plot(kind="bar", color=colors)
        plt.title("Prediction Distribution")
        plt.xlabel("Prediction")
        plt.ylabel("Count")
        plt.xticks([0, 1], ["No Landslide", "Landslide"], rotation=0)
        st.pyplot(plt)  # Display the bar chart in Streamlit

if __name__ == "__main__":
    main()



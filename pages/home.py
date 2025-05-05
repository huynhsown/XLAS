import streamlit as st
import os


def show_home_page(paths):
    """
    Display the home page with app information and setup instructions.

    Args:
        paths: Dictionary containing file paths
    """
    st.markdown("""
    ## Welcome to the Face Recognition System!

    This application provides tools for face detection and recognition using OpenCV and Streamlit.

    ### Features:
    - **Face Recognition from Camera**: Use your webcam for real-time face recognition
    - **Face Recognition from Video**: Upload and analyze videos for face recognition
    - **Manage Face Database**: Add new faces to the recognition database

    ### Setup Instructions:
    1. Place the face detection and recognition models in the 'models' folder
    2. Use the sidebar to navigate between different functionalities
    """)

    # Check if models exist
    detection_exists = os.path.exists(paths["detection_model"])
    recognition_exists = os.path.exists(paths["recognition_model"])

    if detection_exists and recognition_exists:
        st.success("All required models found!")

    # Check if database exists
    database_exists = os.path.exists(paths["database_path"])
    if not database_exists:
        st.info("No face database found. Use the 'Manage Face Database' menu to add faces.")
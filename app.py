import streamlit as st
from pathlib import Path
import os

# Import modules
from config import get_file_paths
from models import load_models
from database import load_database
from pages.home import show_home_page
from pages.show_face_recognize_cam import show_camera_page
from pages.face_recognize_video import show_video_page
from pages.face_data_management import show_database_management_page

st.set_page_config(
    page_title="Face Recognition System",
    page_icon="👤",
    layout="wide"
)

# App title and description
st.title("📊 Face Recognition System")
st.markdown("### A streamlit application for face detection and recognition")


# Create necessary directories
def create_directories():
    base_dir = Path(".")
    models_dir = base_dir / "models"
    data_dir = base_dir / "data"

    # Create directories if they don't exist
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)


# Initialize the app
def main():
    # Create directories
    create_directories()

    # Get file paths
    paths = get_file_paths()

    # Sidebar menu
    st.sidebar.title("Menu")
    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        ["Home", "Face Recognition from Camera", "Face Recognition from Video", "Manage Face Database"]
    )

    # Load models and database
    detector, recognizer = load_models(paths["detection_model"], paths["recognition_model"])
    database = load_database(paths["database_path"])

    # Display appropriate page based on selection
    if app_mode == "Home":
        show_home_page(paths)
    elif app_mode == "Face Recognition from Camera":
        show_camera_page(detector, recognizer, database)
    elif app_mode == "Face Recognition from Video":
        show_video_page(detector, recognizer, database)
    elif app_mode == "Manage Face Database":
        show_database_management_page(detector, recognizer, database, paths["database_path"])


if __name__ == "__main__":
    main()
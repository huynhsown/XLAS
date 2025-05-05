import streamlit as st
from pathlib import Path


# File paths configuration
@st.cache_resource
def get_file_paths():
    """
    Get the file paths for models and database.
    Returns a dictionary with paths.
    """
    base_dir = Path(".")
    models_dir = base_dir / "models"
    data_dir = base_dir / "data"

    # Create directories if they don't exist
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    paths = {
        "detection_model": str(models_dir / "face_detection_yunet_2023mar.onnx"),
        "recognition_model": str(models_dir / "face_recognition_sface_2021dec.onnx"),
        "database_path": str(data_dir / "face_database.pkl")
    }
    return paths
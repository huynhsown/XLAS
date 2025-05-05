import os
import pickle
import streamlit as st


@st.cache_data
def load_database(database_path):
    """
    Load the face database from a pickle file.

    Args:
        database_path: Path to the face database file

    Returns:
        database: Dictionary of face embeddings
    """
    try:
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                return pickle.load(f)
        else:
            return {}
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return {}


def save_database(database, database_path):
    """
    Save the face database to a pickle file.

    Args:
        database: Dictionary of face embeddings
        database_path: Path to save the face database file

    Returns:
        success: Boolean indicating success or failure
    """
    try:
        with open(database_path, 'wb') as f:
            pickle.dump(database, f)
        return True
    except Exception as e:
        st.error(f"Error saving database: {e}")
        return False
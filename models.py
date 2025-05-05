import cv2
import streamlit as st


@st.cache_resource
def load_models(detection_path, recognition_path):
    """
    Load the face detection and recognition models.

    Args:
        detection_path: Path to the face detection model
        recognition_path: Path to the face recognition model

    Returns:
        detector: FaceDetectorYN instance
        recognizer: FaceRecognizerSF instance
    """
    try:
        # Initialize face detector
        detector = cv2.FaceDetectorYN.create(
            detection_path,
            "",
            (320, 320),
            0.9,  # Score threshold
            0.3,  # NMS threshold
            5000  # Top K
        )

        # Initialize face recognizer
        recognizer = cv2.FaceRecognizerSF.create(
            recognition_path,
            ""
        )

        return detector, recognizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None
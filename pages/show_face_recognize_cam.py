import streamlit as st
import cv2
import time
import uuid
from utils.face_processing import process_frame


def show_camera_page(detector, recognizer, database):
    """
    Display the camera recognition page with webcam feed.

    Args:
        detector: FaceDetectorYN instance
        recognizer: FaceRecognizerSF instance
        database: Dictionary of face embeddings
    """
    st.markdown("## Face Recognition from Camera")

    if detector is None or recognizer is None:
        st.error("Failed to load models. Please check model paths.")
        return

    # Camera settings
    camera_options = ["Default Camera (0)"]
    for i in range(1, 5):  # Additional camera options
        camera_options.append(f"Camera {i}")

    selected_camera = st.selectbox("Select Camera", camera_options)
    camera_id = int(selected_camera.split("(")[1].split(")")[0]) if "(" in selected_camera else camera_options.index(
        selected_camera)

    # Recognition threshold
    recognition_threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.4)

    col1, col2 = st.columns(2)
    start_camera = col1.button("Start Camera")

    if start_camera:
        # Create a placeholder for the stop button
        stop_button_placeholder = col2.empty()
        stop_pressed = stop_button_placeholder.button("Stop Camera")

        # Initialize camera
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            st.error(f"Cannot open camera {camera_id}")
            return

        # Create a placeholder for the camera feed
        stframe = st.empty()

        # Create containers for face recognition stats
        stats_container = st.container()
        col1, col2 = stats_container.columns(2)
        faces_detected = col1.empty()
        recognized_faces = col2.empty()

        # Display stats while camera is running
        while not stop_pressed:
            # Read a frame from the camera
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture image")
                break

            # Process the frame for face detection and recognition
            processed_frame, results = process_frame(frame, detector, recognizer, database, recognition_threshold)

            # Update stats
            faces_detected.markdown(f"**Faces Detected:** {len(results)}")
            recognized_count = sum(1 for face in results if face['name'] != "Unknown")
            recognized_faces.markdown(f"**Faces Recognized:** {recognized_count}")

            # Convert to RGB for display in Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display the frame
            stframe.image(processed_frame_rgb, channels="RGB", width=400)

            # Check if stop button is pressed
            stop_pressed = stop_button_placeholder.button("Stop Camera", key=str(uuid.uuid4()))

            # Small delay to reduce CPU usage
            time.sleep(0.01)

        # Release the camera when done
        cap.release()
        stframe.empty()
        st.success("Camera stopped")
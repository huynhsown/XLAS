import streamlit as st
import cv2
import numpy as np
from database import save_database


def show_database_management_page(detector, recognizer, database, database_path):
    """
    Display the database management page for adding/removing faces.

    Args:
        detector: FaceDetectorYN instance
        recognizer: FaceRecognizerSF instance
        database: Dictionary of face embeddings
        database_path: Path to the face database file
    """
    st.markdown("## Manage Face Database")

    if detector is None or recognizer is None:
        st.error("Failed to load models. Please check model paths.")
        return

    # Display current database
    st.subheader("Current Database")
    if database:
        for name in database.keys():
            st.write(f"- {name}")
    else:
        st.write("No faces in database yet.")

    # Create tabs for different operations
    tab1, tab2 = st.tabs(["Add New Face", "Remove Face"])

    # Add new face tab
    with tab1:
        add_new_face(detector, recognizer, database, database_path)

    # Remove face tab
    with tab2:
        remove_face(database, database_path)


def add_new_face(detector, recognizer, database, database_path):
    """
    Interface for adding a new face to the database.

    Args:
        detector: FaceDetectorYN instance
        recognizer: FaceRecognizerSF instance
        database: Dictionary of face embeddings
        database_path: Path to the face database file
    """
    # Name input
    new_face_name = st.text_input("Name")

    # Image upload option
    st.write("Upload a clear frontal face image:")
    uploaded_image = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None and new_face_name:
        # Convert uploaded image to numpy array
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Display the uploaded image
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)

        # Detect face in the image
        h, w = img.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(img)

        if faces is None or len(faces) == 0:
            st.error("No face detected in the image. Please upload a clear image with a face.")
        elif len(faces) > 1:
            st.error("Multiple faces detected. Please upload an image with only one face.")
        else:
            # Extract face and get embedding
            face = faces[0]

            # Draw rectangle around the detected face
            box = list(map(int, face[:4]))
            img_with_face = img.copy()
            cv2.rectangle(img_with_face, (box[0], box[1]),
                          (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)

            # Display the image with detected face
            st.image(cv2.cvtColor(img_with_face, cv2.COLOR_BGR2RGB),
                     caption="Detected Face", use_column_width=True)

            # Get face embedding
            aligned_face = recognizer.alignCrop(img, face)
            face_embedding = recognizer.feature(aligned_face)

            # Add to database
            if st.button("Add to Database"):
                # Check if name already exists
                if new_face_name in database:
                    overwrite = st.checkbox("This name already exists. Overwrite?")
                    if not overwrite:
                        st.warning("Please choose a different name or confirm overwrite.")
                        return

                database[new_face_name] = face_embedding
                if save_database(database, database_path):
                    st.success(f"Added {new_face_name} to the database!")
                    # Refresh the page to reflect changes
                    st.experimental_rerun()


def remove_face(database, database_path):
    """
    Interface for removing a face from the database.

    Args:
        database: Dictionary of face embeddings
        database_path: Path to the face database file
    """
    if database:
        face_to_remove = st.selectbox("Select face to remove", list(database.keys()))
        if st.button("Remove"):
            # Confirm removal
            if st.checkbox("Are you sure? This cannot be undone."):
                database.pop(face_to_remove, None)
                if save_database(database, database_path):
                    st.success(f"Removed {face_to_remove} from the database!")
                    # Refresh the page to reflect changes
                    st.experimental_rerun()
    else:
        st.write("No faces to remove.")
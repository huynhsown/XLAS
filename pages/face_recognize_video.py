import streamlit as st
import cv2
import numpy as np
import uuid
import tempfile
import os
from utils.face_processing import process_frame


def show_video_page(detector, recognizer, database):
    """
    Display the video recognition page for uploaded videos.

    Args:
        detector: FaceDetectorYN instance
        recognizer: FaceRecognizerSF instance
        database: Dictionary of face embeddings
    """
    st.markdown("## Face Recognition from Video")

    if detector is None or recognizer is None:
        st.error("Failed to load models. Please check model paths.")
        return

    # Video upload
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name
        temp_file.close()

        # Open the video file
        cap = cv2.VideoCapture(temp_file_path)

        if not cap.isOpened():
            st.error("Error opening video file")
        else:
            # Video details
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps

            st.write(f"FPS: {fps}, Total Frames: {frame_count}, Duration: {duration:.2f} seconds")

            # Video processing options
            process_every_n_frames = st.slider("Process every N frames", 1, 10, 1)
            recognition_threshold = st.slider("Recognition Threshold", 0.0, 1.0, 0.4)

            # Start processing button
            if st.button("Process Video"):
                # Progress bar and status
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Display area for processed frames
                col1, col2 = st.columns([1, 1])
                frame_display = col1.empty()
                names_display = col2.empty()

                # Results storage
                all_results = []
                frame_index = 0

                # Process the video
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Only process every N frames
                    if frame_index % process_every_n_frames == 0:
                        # Process the frame
                        processed_frame, results = process_frame(
                            frame, detector, recognizer, database, recognition_threshold
                        )

                        # Store results
                        all_results.append({
                            "frame": frame_index,
                            "time": frame_index / fps,
                            "faces": results
                        })

                        # Display the frame
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        # Hiển thị ảnh
                        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        frame_display.image(frame_rgb, channels="RGB", width=400)

                        # Hiển thị danh sách tên nhận diện được
                        recognized_names = [face["name"] for face in results if face["name"] != "Unknown"]
                        if recognized_names:
                            unique_names = sorted(set(recognized_names))
                            names_display.markdown("### Recognized:")
                            names_display.markdown("\n".join(f"- **{name}**" for name in unique_names))
                        else:
                            names_display.markdown("### Recognized:\n_Unknown_")

                        # Update progress
                        progress = (frame_index + 1) / frame_count
                        progress_bar.progress(progress)
                        status_text.text(f"Processing frame {frame_index + 1}/{frame_count} ({progress * 100:.1f}%)")

                    frame_index += 1

                # Clean up
                cap.release()
                os.unlink(temp_file_path)

                # Display summary
                st.success("Video processing completed")

                # Show results
                show_video_results(all_results)


def show_video_results(all_results):
    """
    Display video processing results.

    Args:
        all_results: List of results from video processing
    """
    # Extract unique faces and count appearances
    unique_faces = set()
    face_appearances = {}
    face_timestamps = {}

    for frame_result in all_results:
        time_in_video = frame_result["time"]
        for face in frame_result["faces"]:
            if face["name"] != "Unknown":
                unique_faces.add(face["name"])
                face_appearances[face["name"]] = face_appearances.get(face["name"], 0) + 1

                # Track timestamps
                if face["name"] not in face_timestamps:
                    face_timestamps[face["name"]] = []
                face_timestamps[face["name"]].append(time_in_video)

    st.subheader("Recognition Results")
    st.write(f"Detected {len(unique_faces)} unique individuals")

    if unique_faces:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Summary", "Timeline"])

        # Summary tab
        with tab1:
            # Display faces and their appearance counts
            for name in unique_faces:
                appearances = face_appearances.get(name, 0)
                st.write(f"- **{name}**: appeared in {appearances} frames")

        # Timeline tab
        with tab2:
            # Show when each person appeared in the video
            for name in unique_faces:
                timestamps = face_timestamps.get(name, [])
                if timestamps:
                    first_appearance = min(timestamps)
                    last_appearance = max(timestamps)
                    st.write(f"- **{name}**: First appeared at {first_appearance:.2f}s, "
                             f"Last seen at {last_appearance:.2f}s")
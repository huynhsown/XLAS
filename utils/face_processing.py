import cv2


def process_frame(frame, detector, recognizer, database, threshold=0.4):
    """
    Process a frame to detect and recognize faces.

    Args:
        frame: Input frame/image
        detector: FaceDetectorYN instance
        recognizer: FaceRecognizerSF instance
        database: Dictionary of face embeddings
        threshold: Recognition threshold (default: 0.4)

    Returns:
        processed_frame: Frame with annotations
        results: List of detected faces with recognition results
    """
    results = []

    # Detect faces
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)

    if faces is not None:
        for face in faces:
            # Extract face information
            box = list(map(int, face[:4]))
            confidence = face[14]

            # Only process faces with good confidence
            if confidence > 0.9:
                # Extract embedding
                aligned_face = recognizer.alignCrop(frame, face)
                query_embedding = recognizer.feature(aligned_face)

                # Compare with database
                max_score = 0
                best_match = "Unknown"
                if database:
                    for ref_name, ref_embedding in database.items():
                        score = recognizer.match(query_embedding, ref_embedding, cv2.FaceRecognizerSF_FR_COSINE)
                        if score > max_score:
                            max_score = score
                            best_match = ref_name

                # Only label as known if score is high enough
                if max_score < threshold:
                    best_match = "Unknown"

                # Draw rectangle and label
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                label = f"{best_match} ({max_score:.2f})" if max_score > 0 else "Unknown"
                cv2.putText(frame, label, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store result
                results.append({
                    "box": box,
                    "name": best_match,
                    "score": max_score,
                    "embedding": query_embedding
                })

    return frame, results
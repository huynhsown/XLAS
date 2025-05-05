import cv2
import numpy as np
from collections import defaultdict
import pickle
import os

# Đường dẫn đến model
detection_model = "D:/MyWorkSpace/XLAS/models/face_detection_yunet_2023mar.onnx"
recognition_model = "D:/MyWorkSpace/XLAS/models/face_recognition_sface_2021dec.onnx"

# Đường dẫn lưu database
database_path = "D:/MyWorkSpace/XLAS/data/face_database.pkl"

# Khởi tạo face detector và recognizer
detector = cv2.FaceDetectorYN.create(
    detection_model,
    "",
    (320, 320),
    0.9,  # Score threshold
    0.3,  # NMS threshold
    5000  # Top K
)

recognizer = cv2.FaceRecognizerSF.create(
    recognition_model,
    ""
)

# Danh sách video
video_files = {
    "dang_huynh_son": "D:/MyWorkSpace/XLAS/data/video_face_data/dang_huynh_son.mp4",
    "hoang_huy": "D:/MyWorkSpace/XLAS/data/video_face_data/hoang_huy.mp4",
    "nguyen_hoang_phuc": "D:/MyWorkSpace/XLAS/data/video_face_data/nguyen_hoang_phuc.mp4",
    "van_luan": "D:/MyWorkSpace/XLAS/data/video_face_data/van_luan.mp4",
    "trung_ky": "D:/MyWorkSpace/XLAS/data/video_face_data/trung_ky.mp4"
}

# Kiểm tra nếu database đã tồn tại thì load lại, không thì tạo mới
if os.path.exists(database_path):
    print("Đang tải database từ file...")
    with open(database_path, 'rb') as f:
        avg_reference = pickle.load(f)
else:
    print("Đang xây dựng database mới...")
    # Bước 1: Xây dựng database khuôn mặt từ các video
    reference_embeddings = defaultdict(list)

    for name, video_path in video_files.items():
        print(f"Đang xử lý video tham chiếu cho {name}...")
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect khuôn mặt
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(frame)

            if faces is not None:
                for face in faces:
                    aligned_face = recognizer.alignCrop(frame, face)
                    embedding = recognizer.feature(aligned_face)
                    reference_embeddings[name].append(embedding)

        cap.release()

    avg_reference = {}
    for name, embeddings in reference_embeddings.items():
        avg_reference[name] = np.mean(embeddings, axis=0)

    # Lưu database vào file
    with open(database_path, 'wb') as f:
        pickle.dump(avg_reference, f)
    print("Đã lưu database vào file.")

# Phần nhận diện trong video (giữ nguyên)
for name, video_path in video_files.items():
    print(f"Đang nhận diện trong video {name}...")
    cap = cv2.VideoCapture(video_path)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect khuôn mặt
        h, w = frame.shape[:2]
        detector.setInputSize((w, h))
        _, faces = detector.detect(frame)

        if faces is not None:
            for face in faces:
                # Trích xuất embedding
                aligned_face = recognizer.alignCrop(frame, face)
                query_embedding = recognizer.feature(aligned_face)

                # So sánh với database
                max_score = 0
                best_match = "Unknown"
                for ref_name, ref_embedding in avg_reference.items():
                    score = recognizer.match(query_embedding, ref_embedding, cv2.FaceRecognizerSF_FR_COSINE)
                    if score > max_score:
                        max_score = score
                        best_match = ref_name

                # Vẽ kết quả
                box = list(map(int, face[:4]))
                cv2.rectangle(frame, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{best_match} ({max_score:.2f})", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Hiển thị frame
        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
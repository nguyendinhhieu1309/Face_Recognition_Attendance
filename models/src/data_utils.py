import os
import cv2

def create_processed_data_directory():
    processed_directory = "F:\SUMMER24\CPV301\Face_Recognition_Attendance_System\processed_data"
    if not os.path.exists(processed_directory):
        os.makedirs(processed_directory)
        print(f"Created directory: {processed_directory}")
    else:
        print(f"Directory already exists: {processed_directory}")

def capture_student_images(user_id):
    url = 'http://192.168.0.100:8080/video'  # Thay thế bằng thông số của camera IP
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print(f"Error opening video stream with URL: {url}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_num = 0
    interval = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w]
            roi_color = cv2.resize(crop_img, (160, 160))

            if face_num % interval == 0:
                img_directory = os.path.join('F:\SUMMER24\CPV301\Face_Recognition_Attendance_System\processed_data', user_id)
                img_filename = os.path.join(img_directory, f'frame_{face_num}.jpg')
                if not os.path.exists(img_directory):
                    os.makedirs(img_directory)
                cv2.imwrite(img_filename, roi_color)

            face_num += 1

        cv2.namedWindow('Capture', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Capture', 640, 480)
        cv2.imshow('Capture', frame)

        if face_num == 50:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    
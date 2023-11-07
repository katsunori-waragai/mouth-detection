"""
mediapipe を用いて顔のメッシュを描画するサンプルスクリプト
# https://yoppa.org/mit-design4-22/14113.html

https://github.com/google/mediapipe/issues/1020
"""

import cv2
import mediapipe as mp

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="face_mash")
    parser.add_argument("video", help="video_file")
    args = parser.parse_args()

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh

    # Webカメラから入力
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    if args.video.find("/dev/video") == 0:
        video_num = int(args.video.replace("/dev/video", ""))
        cap = cv2.VideoCapture(video_num)
    else:
        cap = cv2.VideoCapture(args.video)
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image)

            # 検出された顔のメッシュをカメラ画像の上に描画
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style(),
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_IRISES,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                    )
            print(face_landmarks.landmark)
            landmarks = [{
                 'x': data_point.x,
                 'y': data_point.y,
                 'z': data_point.z,
                 'Visibility': data_point.visibility,
             } for data_point in face_landmarks.landmark]
            print(f"{type(landmarks)=}")
            points = [landmarks[i] for i in (0, 3, 14, 17, 57, 287)]
            xmin = min((p["x"] for p in points))
            xmax = max((p["x"] for p in points))
            ymin = min((p["y"] for p in points))
            ymax = max((p["y"] for p in points))
            print(f"{xmin} {ymin} {xmax} {ymax}")
            # cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0,255,0), 3)
            cv2.imshow("MediaPipe Face Mesh", cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

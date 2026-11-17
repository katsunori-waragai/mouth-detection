"""
mediapipe を用いて顔のメッシュを描画するサンプルスクリプト
# 雛形にしたコード
# https://yoppa.org/mit-design4-22/14113.html

mediapipe の各landmarkの位置
https://storage.googleapis.com/mediapipe-assets/documentation/mediapipe_face_landmark_fullsize.png

Extract the landmarks of pose tracker in python
https://github.com/google/mediapipe/issues/1020
"""
import argparse
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class FaceInfo:
    """顔ランドマークから抽出した口領域の情報を保持するデータクラス。

    Attributes:
        landmarks: 正規化された顔ランドマークのリスト。
        box: 口領域の外接矩形 (xmin, ymin, xmax, ymax)。
        mouth_ratio: 口の開口度を示す比率。
    """

    landmarks: List[dict]
    box: Tuple[int, int, int, int]
    mouth_ratio: float


class FaceMeshProcessor:
    """MediaPipe の顔メッシュ検出処理をラップするクラス。

    このクラスはフレームから顔ランドマークを抽出し、他のクラスが
    解析しやすい形式で結果を返す役割を担う。
    """
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, image: np.ndarray):
        image.flags.writeable = False
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        image.flags.writeable = True
        return results

    def close(self) -> None:
        self.face_mesh.close()


class MouthAnalyzer:
    """顔ランドマークから口の位置と開口度を計算するクラス。

    口の動作分析では、特定のランドマークの位置関係から口の矩形と
    開き具合を定量化する。
    """

    KEY_POINTS = (0, 13, 14, 17, 57, 287)

    @staticmethod
    def normalize_landmarks(face_landmarks) -> List[dict]:
        return [
            {
                "x": data_point.x,
                "y": data_point.y,
                "z": data_point.z,
                "visibility": getattr(data_point, "visibility", None),
            }
            for data_point in face_landmarks.landmark
        ]

    @staticmethod
    def get_corner_points(points: List[dict], image_shape) -> Tuple[int, int, int, int]:
        x_values = [p["x"] for p in points]
        y_values = [p["y"] for p in points]

        xmin = int(min(x_values) * image_shape[1])
        xmax = int(max(x_values) * image_shape[1])
        ymin = int(min(y_values) * image_shape[0])
        ymax = int(max(y_values) * image_shape[0])
        return xmin, ymin, xmax, ymax

    def build_face_info(self, face_landmarks, image_shape) -> Optional[FaceInfo]:
        landmarks = self.normalize_landmarks(face_landmarks)
        points = [landmarks[i] for i in self.KEY_POINTS]
        xmin, ymin, xmax, ymax = self.get_corner_points(points, image_shape)

        if xmax <= xmin:
            return None

        y13 = int(landmarks[13]["y"] * image_shape[0])
        y14 = int(landmarks[14]["y"] * image_shape[0])
        mouth_ratio = (y14 - y13) / (xmax - xmin)

        return FaceInfo(
            landmarks=landmarks,
            box=(xmin, ymin, xmax, ymax),
            mouth_ratio=mouth_ratio,
        )


class FaceMeshVisualizer:
    """検出した顔情報を OpenCV で描画するクラス。

    顔メッシュの線、特定ポイント、口領域の矩形、開口度テキストを
    画像に重ねて表示する。
    """

    def __init__(self) -> None:
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face_mesh = mp.solutions.face_mesh

    def draw_face_mesh(self, image: np.ndarray, face_landmarks) -> None:
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        )
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style(),
        )
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
        )

    def draw_face_info(self, image: np.ndarray, face_info: FaceInfo) -> None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (128, 128, 0), (0, 128, 128), (0, 0, 128)]
        points = [face_info.landmarks[i] for i in MouthAnalyzer.KEY_POINTS]

        for j, p in enumerate(points):
            x = int(p["x"] * image.shape[1])
            y = int(p["y"] * image.shape[0])
            cv2.circle(image, (x, y), 5, color=colors[j % 6], thickness=3)
            cv2.putText(
                image,
                f"{j}",
                org=(x, y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=colors[j % 6],
                thickness=2,
            )

        xmin, ymin, xmax, ymax = face_info.box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        cv2.putText(
            image,
            f"ratio={face_info.mouth_ratio:.2f}",
            org=(xmin, max(0, ymin - 10)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=2,
        )


class FaceMeshApp:
    """動画またはカメラ入力を処理して顔メッシュ解析を実行するアプリ本体。

    Args:
        source: 入力元の動画ファイルパスまたはカメラデバイスパス。
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.processor = FaceMeshProcessor()
        self.analyzer = MouthAnalyzer()
        self.visualizer = FaceMeshVisualizer()

    def open_capture(self):
        if self.source.startswith("/dev/video"):
            video_num = int(self.source.replace("/dev/video", ""))
            return cv2.VideoCapture(video_num)
        return cv2.VideoCapture(self.source)

    def run(self) -> None:
        cap = self.open_capture()
        cv2.namedWindow("MediaPipe Face Mesh", cv2.WINDOW_NORMAL)

        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                results = self.processor.process(image)
                if results and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        self.visualizer.draw_face_mesh(image, face_landmarks)
                        face_info = self.analyzer.build_face_info(face_landmarks, image.shape)
                        if face_info is not None:
                            self.visualizer.draw_face_info(image, face_info)

                cv2.imshow("MediaPipe Face Mesh", image)
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.processor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="face mesh")
    parser.add_argument("video", help="video file or /dev/video0")
    args = parser.parse_args()

    app = FaceMeshApp(args.video)
    app.run()


if __name__ == "__main__":
    main()


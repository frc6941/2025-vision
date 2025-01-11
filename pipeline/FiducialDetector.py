import cv2

from config.config import ConfigStore
from vision_types import FiducialImageObservation


class FiducialDetector:
    def __init__(self) -> None:
        pass

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> list[FiducialImageObservation]:
        raise NotImplementedError


class ArucoFiducialDetector(FiducialDetector):
    def __init__(self, dictionary_id) -> None:
        super().__init__()
        self._aruco_detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary_id),
                                                       cv2.aruco.DetectorParameters())

    def detect_fiducials(self, image: cv2.Mat, config_store: ConfigStore) -> list[FiducialImageObservation]:
        corners, ids, _ = self._aruco_detector.detectMarkers(image)
        if len(corners) == 0:
            return []
        return [FiducialImageObservation(i[0], corner) for i, corner in zip(ids, corners)]

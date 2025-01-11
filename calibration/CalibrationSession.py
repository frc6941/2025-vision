import datetime
import os

import cv2
import numpy

from config.ConfigSource import FileConfigSource


class CalibrationSession:

    def __init__(self) -> None:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
        self._aruco_detector = cv2.aruco.ArucoDetector(aruco_dict,
                                                       cv2.aruco.DetectorParameters())

        self._charuco_board = cv2.aruco.CharucoBoard(
            (12, 9), 0.030, 0.023, aruco_dict)
        self._charuco_detector = cv2.aruco.CharucoDetector(self._charuco_board)

        self._all_charuco_corners: list[numpy.ndarray] = []
        self._all_charuco_ids: list[numpy.ndarray] = []
        self._imsize = None
        self._picture_cnt: int = 0

    def process_frame(self, image: cv2.Mat, save: bool) -> None:
        # Get image size
        if self._imsize is None:
            self._imsize = (image.shape[0], image.shape[1])

        # Detect tags
        corners, ids, rejected = self._aruco_detector.detectMarkers(image)
        if len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(image, corners)

            # Find Charuco corners
            retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                corners, ids, image, self._charuco_board)
            if retval:
                cv2.aruco.drawDetectedCornersCharuco(image, charuco_corners, charuco_ids)

                # Save corners
                if save:
                    self._picture_cnt += 1
                    self._all_charuco_corners.append(charuco_corners)
                    self._all_charuco_ids.append(charuco_ids)
                    print("Saved, %d calibration frame(s) in total" % self._picture_cnt)

    def finish(self) -> None:
        if len(self._all_charuco_corners) == 0:
            print("ERROR: No calibration data")
            return

        if os.path.exists(FileConfigSource.CALIBRATION_FILENAME):
            os.remove(FileConfigSource.CALIBRATION_FILENAME)

        # noinspection PyTypeChecker
        retval, camera_matrix, distortion_coefficients, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            self._all_charuco_corners, self._all_charuco_ids, self._charuco_board, self._imsize, None, None)

        if retval:
            calibration_store = cv2.FileStorage(FileConfigSource.CALIBRATION_FILENAME, cv2.FILE_STORAGE_WRITE)
            calibration_store.write("calibration_date", str(datetime.datetime.now()))
            calibration_store.write("camera_resolution", self._imsize)
            calibration_store.write("camera_matrix", camera_matrix)
            calibration_store.write("distortion_coefficients", distortion_coefficients)
            calibration_store.release()
            print("Calibration finished")
        else:
            print("ERROR: Calibration failed")

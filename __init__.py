import multiprocessing
import sys
import time
from multiprocessing import cpu_count, Pool, Manager
from typing import Union

import ntcore

from calibration.CalibrationCommandSource import (CalibrationCommandSource,
                                                  NTCalibrationCommandSource)
from calibration.CalibrationSession import CalibrationSession
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from config.config import LocalConfig, RemoteConfig
from output.OutputPublisher import NTOutputPublisher, OutputPublisher
from output.StreamServer import MjpegServer
from output.overlay_util import *
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.Capture import DefaultCapture
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator

DEMO_ID = 29


def get_image_observations(q_in, q_out):
    while True:
        if not q_in.empty():
            image = q_in.get()
            config = ConfigStore(LocalConfig(), RemoteConfig())
            img = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11).detect_fiducials(image, config)
            q_out.put(img)


if __name__ == "__main__":
    # multiprocessing to speed up
    pool = Pool(cpu_count())
    queueIn: multiprocessing.Queue = Manager().Queue(maxsize=cpu_count)
    queueOut = Manager().Queue(maxsize=cpu_count)
    for i in range(cpu_count()):
        pool.apply_async(func=get_image_observations, args=(queueIn, queueOut))

    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()

    # capture = GStreamerCapture()  # linux only
    capture = DefaultCapture()  # other platforms
    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    tag_pose_estimator = SquareTargetPoseEstimator()
    output_publisher: OutputPublisher = NTOutputPublisher()
    stream_server = MjpegServer()
    calibration_session = CalibrationSession()

    local_config_source.update(config)
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)
    stream_server.start(config)

    frame_count = 0
    last_print = 0
    was_calibrating = False
    while True:
        remote_config_source.update(config)
        timestamp = time.time()
        success, image = capture.get_frame(config)
        if not success:
            time.sleep(0.5)
            continue

        fps = None
        frame_count += 1
        if time.time() - last_print > 1:
            last_print = time.time()
            fps = frame_count
            print("Running at", frame_count, "fps")
            frame_count = 0

        if calibration_command_source.get_calibrating(config):
            # Calibration mode
            was_calibrating = True
            calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))

        elif was_calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0)

        elif config.local_config.has_calibration:
            # Normal mode
            while not queueIn.empty():
                success, image = capture.get_frame(config)
                if not success:
                    time.sleep(0.5)
                    continue
                queueIn.put(image)
            while not queueOut.empty():
                image_observations = queueOut.get()
                [overlay_image_observation(image, x) for x in image_observations]
                camera_pose_observation = camera_pose_estimator.solve_camera_pose(
                    [x for x in image_observations if x.tag_id != DEMO_ID], config)
                demo_image_observations = [x for x in image_observations if x.tag_id == DEMO_ID]
                demo_pose_observation: Union[FiducialPoseObservation, None] = None
                if len(demo_image_observations) > 0:
                    demo_pose_observation = tag_pose_estimator.solve_fiducial_pose(demo_image_observations[0], config)
                output_publisher.send(config, timestamp, camera_pose_observation, demo_pose_observation, fps)

        else:
            # No calibration
            print("No calibration found")
            time.sleep(0.5)

        # image = cv2.undistort(image, config.local_config.camera_matrix, config.local_config.distortion_coefficients)
        stream_server.set_frame(image)

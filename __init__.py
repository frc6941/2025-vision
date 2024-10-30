import multiprocessing
import sys
import time
from multiprocessing import cpu_count
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


def imgProcesser(qImage: multiprocessing.Queue, qTime: multiprocessing.Queue, qConfig: multiprocessing.Queue,
                 qResult: multiprocessing.Queue, fps_count):
    output_publisher: OutputPublisher = NTOutputPublisher()
    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
    # estimator
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    tag_pose_estimator = SquareTargetPoseEstimator()
    while True:
        if not qImage.empty():
            image = qImage.get()
            pTime = qTime.get()
            pConfig = qConfig.get()
            image_observations = fiducial_detector.detect_fiducials(image, pConfig)
            [overlay_image_observation(image, x) for x in image_observations]
            camera_pose_observation = camera_pose_estimator.solve_camera_pose(
                [x for x in image_observations if x.tag_id != DEMO_ID], pConfig)
            demo_image_observations = [x for x in image_observations if x.tag_id == DEMO_ID]
            demo_pose_observation: Union[FiducialPoseObservation, None] = None
            if len(demo_image_observations) > 0:
                demo_pose_observation = tag_pose_estimator.solve_fiducial_pose(demo_image_observations[0], pConfig)
            output_publisher.send(pConfig, pTime, camera_pose_observation, demo_pose_observation, fps_count.value)
            qResult.put(image)


def imgPublisher(qImage: multiprocessing.Queue, qTime: multiprocessing.Queue, qConfig: multiprocessing.Queue,
                 fps_count):
    capture = DefaultCapture()
    config = ConfigStore(LocalConfig(), RemoteConfig())
    remote_config_source: ConfigSource = NTConfigSource()
    while True:
        # update config
        remote_config_source.update(config)

        # get image
        success, image = capture.get_frame(config)
        if not success:
            print("Failed to get image")
            time.sleep(0.5)

        # publish image with timestamp & config if processes aren't working
        elif qImage.empty():
            qImage.put(image)
            qTime.put(time.time())
            qConfig.put(config)
            fps_count.value += 1


def streaming(qResult: multiprocessing.Queue):
    print(11)
    config = ConfigStore(LocalConfig(), RemoteConfig())
    # start stream server
    stream_server = MjpegServer()
    stream_server.start(config)
    while True:
        if not qResult.empty():
            stream_server.set_frame(qResult.get())


if __name__ == "__main__":
    capture = DefaultCapture()

    # multiprocessing to speed up
    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool(processes=cpu_count() - 3)
    pool2 = multiprocessing.Pool(processes=1)
    pool3 = multiprocessing.Pool(processes=1)

    # variables sharing between processes
    queue_image = manager.Queue()
    queue_time = manager.Queue()
    queue_config = manager.Queue()
    queue_result = manager.Queue()
    fps_count = manager.Value('i', 0)

    # create cpu_count() process
    for i in range(cpu_count() - 3):
        pool.apply_async(func=imgProcesser, args=(queue_image, queue_time, queue_config, queue_result, fps_count))
    pool2.apply_async(func=imgPublisher, args=(queue_image, queue_time, queue_config, fps_count))
    pool3.apply_async(func=streaming, args=(queue_result,))

    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()

    # calibration setting
    calibration_session = CalibrationSession()

    # start NT4
    local_config_source.update(config)
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)

    last_print = 0

    while True:
        # update config
        remote_config_source.update(config)

        # print fps
        if time.time() - last_print > 1:
            last_print = time.time()
            print("Running at", fps_count.value, "fps")
            fps_count.value = 0

        # check if calibrating
        while calibration_command_source.get_calibrating(config):
            # terminate processes to get image in main process
            pool.terminate()
            while True:
                # get image
                success, image = capture.get_frame(config)
                while not success:
                    print("Failed to get image")
                    time.sleep(0.5)
                    success, image = capture.get_frame(config)

                # calib
                calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))

                # Finish calibration
                if not calibration_command_source.get_calibrating(config):
                    calibration_session.finish()
                    sys.exit(0)

        if not config.local_config.has_calibration:
            print("No calibration found")
            time.sleep(0.5)

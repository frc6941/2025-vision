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
from vision_types import CameraPoseObservation, FiducialPoseObservation

DEMO_ID = 29


def imgProcessor(qImage: multiprocessing.Queue, qTime: multiprocessing.Queue, qConfig: multiprocessing.Queue,
                 qResult: multiprocessing.Queue, qObservationResult: multiprocessing.Queue, fps_count):
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
            print(22)
            # a = DetectResult(config=pConfig, time=pTime, observation=camera_pose_observation,
            #                  demo_observation=demo_pose_observation,
            #                  fps_count=fps_count.value)
            # a = DetectResult()
            # a.fps_count = fps_count.value
            # a.config = pConfig
            # a.observation = camera_pose_observation
            # a.demo_observation = demo_pose_observation
            # a.time = pTime
            # print(fps_count.value, pConfig, camera_pose_observation, demo_pose_observation, pTime)
            a = DetectResult(fps_count.value, pConfig, camera_pose_observation, demo_pose_observation, pTime)
            qObservationResult.put(a)
            qResult.put(image)
        print(33)


def streaming(qResult: multiprocessing.Queue, fps_count):
    config = ConfigStore(LocalConfig(), RemoteConfig())
    # start stream server
    stream_server = MjpegServer()
    stream_server.start(config)
    # show 1 frame every display_freq frames
    # TODO: remote config
    cnt = 0
    display_freq = 5
    while True:
        if not qResult.empty():
            cnt += 1
            if cnt % display_freq == 0:
                stream_server.set_frame(qResult.get())
                cnt = 0
            else:
                qResult.get()
            fps_count.value += 1
        # else:
        #     print(111)


if __name__ == "__main__":
    capture = DefaultCapture()

    # multiprocessing to speed up
    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool(processes=cpu_count() - 2)
    pool2 = multiprocessing.Pool(processes=1)

    # variables sharing between processes
    queue_image = manager.Queue()
    queue_time = manager.Queue()
    queue_config = manager.Queue()
    queue_result = manager.Queue()
    queue_observation_result = manager.Queue()
    fps_count = manager.Value('i', 0)

    # create cpu_count() process
    for i in range(cpu_count() - 2):
        pool.apply_async(func=imgProcessor, args=(
            queue_image, queue_time, queue_config, queue_result, queue_observation_result, fps_count))
    pool2.apply_async(func=streaming, args=(queue_result, fps_count))

    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()

    # calibration setting
    calibration_session = CalibrationSession()
    calibrating = False

    # start NT4
    local_config_source.update(config)
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)

    # fps_counting
    last_print = 0

    # publish output
    output_publisher: OutputPublisher = NTOutputPublisher()

    while True:
        # update config
        remote_config_source.update(config)

        # print fps
        if time.time() - last_print > 1:
            last_print = time.time()
            print("Running at", fps_count.value, "fps")
            fps_count.value = 0

        # get image
        success, image = capture.get_frame(config)
        while not success:
            print("Failed to get image")
            time.sleep(0.5)
            success, image = capture.get_frame(config)

        # publish image with timestamp & config if processes aren't working
        if queue_image.empty():
            queue_image.put(image)
            queue_time.put(time.time())
            queue_config.put(config)

        # publish result
        if not queue_observation_result.empty():
            print(44)
            observation_result = queue_observation_result.get()
            output_publisher.send(observation_result.config, observation_result.time, observation_result.observation,
                                  observation_result.demo_observation, observation_result.fps_count)

        # Start calibration
        if calibration_command_source.get_calibrating(config):
            if not calibrating:
                # terminate processes to get image in main process
                pool.terminate()
                calibrating = True
            # calib
            calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))

        elif calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0)

        if not config.local_config.has_calibration:
            print("No calibration found")
            time.sleep(0.5)


class DetectResult:
    def __init__(self, config: ConfigStore,
                 time: float,
                 observation: Union[CameraPoseObservation, None],
                 demo_observation: Union[FiducialPoseObservation, None],
                 fps_count: int):
        self.config: ConfigStore = config
        self.time: float = time
        self.observation: Union[CameraPoseObservation, None] = observation
        self.demo_observation: Union[FiducialPoseObservation, None] = demo_observation
        self.fps_count: int = fps_count

    # def __init__(self):
    #     self.config: ConfigStore
    #     self.time: float
    #     self.observation: Union[CameraPoseObservation, None]
    #     self.demo_observation: Union[FiducialPoseObservation, None]
    #     self.fps_count: int

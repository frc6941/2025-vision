import math
import multiprocessing
import sys
import time
from multiprocessing import cpu_count
from typing import Union, List

import ntcore

from calibration.CalibrationCommandSource import (
    CalibrationCommandSource,
    NTCalibrationCommandSource,
)
from calibration.CalibrationSession import CalibrationSession
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from config.config import LocalConfig, RemoteConfig
from output.DetectResult import DetectResult
from output.StreamServer import MjpegServer
from output.overlay_util import *
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.Capture import GStreamerCapture
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from vision_types import FiducialPoseObservation, CameraPoseObservation


def imgProcessor(
        qImage,
        qTime,
        qConfig,
        qResult,
        qDetection,
        fps_count,
):
    DEMO_ID = 29
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
                [x for x in image_observations if x.tag_id != DEMO_ID], pConfig
            )
            demo_image_observations = [
                x for x in image_observations if x.tag_id == DEMO_ID
            ]
            demo_pose_observation: Union[FiducialPoseObservation, None] = None
            if len(demo_image_observations) > 0:
                demo_pose_observation = tag_pose_estimator.solve_fiducial_pose(
                    demo_image_observations[0], pConfig
                )
            send(
                qDetection=qDetection,
                timestamp=pTime,
                observation=camera_pose_observation,
                demo_observation=demo_pose_observation,
                fps=fps_count.value,
            )
            qResult.put(image)


def streaming(qResult, fps_count):
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


def send(
        qDetection,
        timestamp: float,
        observation: Union[CameraPoseObservation, None],
        demo_observation: Union[FiducialPoseObservation, None],
        fps: Union[int, None] = None,
) -> None:
    # print(time.time())
    observation_data: List[float] = [0]
    demo_observation_data: List[float] = []
    if observation != None:
        # print(observation)
        observation_data[0] = 1
        observation_data.append(observation.error_0)
        observation_data.append(observation.pose_0.translation().X())
        observation_data.append(observation.pose_0.translation().Y())
        observation_data.append(observation.pose_0.translation().Z())
        observation_data.append(observation.pose_0.rotation().getQuaternion().W())
        observation_data.append(observation.pose_0.rotation().getQuaternion().X())
        observation_data.append(observation.pose_0.rotation().getQuaternion().Y())
        observation_data.append(observation.pose_0.rotation().getQuaternion().Z())
        if observation.error_1 != None and observation.pose_1 != None:
            observation_data[0] = 2
            observation_data.append(observation.error_1)
            observation_data.append(observation.pose_1.translation().X())
            observation_data.append(observation.pose_1.translation().Y())
            observation_data.append(observation.pose_1.translation().Z())
            observation_data.append(
                observation.pose_1.rotation().getQuaternion().W()
            )
            observation_data.append(
                observation.pose_1.rotation().getQuaternion().X()
            )
            observation_data.append(
                observation.pose_1.rotation().getQuaternion().Y()
            )
            observation_data.append(
                observation.pose_1.rotation().getQuaternion().Z()
            )
        for tag_id in observation.tag_ids:
            observation_data.append(tag_id)
    if demo_observation != None:
        demo_observation_data.append(demo_observation.error_0)
        demo_observation_data.append(demo_observation.pose_0.translation().X())
        demo_observation_data.append(demo_observation.pose_0.translation().Y())
        demo_observation_data.append(demo_observation.pose_0.translation().Z())
        demo_observation_data.append(
            demo_observation.pose_0.rotation().getQuaternion().W()
        )
        demo_observation_data.append(
            demo_observation.pose_0.rotation().getQuaternion().X()
        )
        demo_observation_data.append(
            demo_observation.pose_0.rotation().getQuaternion().Y()
        )
        demo_observation_data.append(
            demo_observation.pose_0.rotation().getQuaternion().Z()
        )
        demo_observation_data.append(demo_observation.error_1)
        demo_observation_data.append(demo_observation.pose_1.translation().X())
        demo_observation_data.append(demo_observation.pose_1.translation().Y())
        demo_observation_data.append(demo_observation.pose_1.translation().Z())
        demo_observation_data.append(
            demo_observation.pose_1.rotation().getQuaternion().W()
        )
        demo_observation_data.append(
            demo_observation.pose_1.rotation().getQuaternion().X()
        )
        demo_observation_data.append(
            demo_observation.pose_1.rotation().getQuaternion().Y()
        )
        demo_observation_data.append(
            demo_observation.pose_1.rotation().getQuaternion().Z()
        )
    # return fps, observation_data, demo_observation_data, math.floor(timestamp * 1000000)
    qDetection.put(DetectResult(fps, observation_data, demo_observation_data, math.floor(timestamp * 1000000)))


def imgPublisher(qImage, qTime, qConfig):
    # capture = DefaultCapture()
    capture = GStreamerCapture()
    config = ConfigStore(LocalConfig(), RemoteConfig())
    remote_config_source: ConfigSource = NTConfigSource()
    while True:
        time_start = time.time()
        # update config
        remote_config_source.update(config)

        # get image
        success, image = capture.get_frame(config)
        while not success:
            print("Failed to get image")
            time.sleep(0.5)

        # publish image with timestamp & config if processes aren't working
        if qImage.empty():
            print("Image Before Queue "+str(time.time()-time_start))
            time1 = time.time()
            qImage.put(image)
            print("Queue " + str(time.time() - time1))
            qTime.put(time.time())
            qConfig.put(config)
            # cv2.imshow("a", image)
            # cv2.waitKey(1)
            print("Get Image: " + str(time.time() - time_start)+"\n")


if __name__ == "__main__":

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
    queue_detection = manager.Queue()
    fps_count = manager.Value("i", 0)

    # create cpu_count() process
    for i in range(cpu_count() - 3):  # TODO: change range when commit
        pool.apply_async(
            func=imgProcessor,
            args=(
                queue_image,
                queue_time,
                queue_config,
                queue_result,
                queue_detection,
                fps_count,
            ),
        )
    pool2.apply_async(func=streaming, args=(queue_result, fps_count,), )
    pool3.apply_async(func=imgPublisher, args=(queue_image, queue_time, queue_config,), )

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

    # NT
    _demo_observations_pub: ntcore.DoubleArrayPublisher
    _observations_pub: ntcore.DoubleArrayPublisher
    _fps_pub: ntcore.IntegerPublisher
    config = ConfigStore(LocalConfig(), RemoteConfig())
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)
    nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
        "/" + config.local_config.device_id + "/output"
    )
    _observations_pub = nt_table.getDoubleArrayTopic(
        "observations"
    ).publish(
        ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True)
    )
    _demo_observations_pub = nt_table.getDoubleArrayTopic(
        "demo_observations"
    ).publish(
        ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True)
    )
    _fps_pub = nt_table.getIntegerTopic("fps").publish()

    while True:
        # update config
        remote_config_source.update(config)

        # print fps
        if time.time() - last_print > 1:
            last_print = time.time()
            print("Running at", fps_count.value, "fps")
            fps_count.value = 0

        if not queue_detection.empty():
            result: DetectResult = queue_detection.get()
            _fps_pub.set(result.fps)
            _observations_pub.set(result.observation, result.time)
            _demo_observations_pub.set(
                result.demo_observation, result.time)
            ntcore.NetworkTableInstance.getDefault().flush()

        # Start calibration
        if calibration_command_source.get_calibrating(config):
            if not calibrating:
                # terminate processes to get image in main process
                pool.terminate()
                calibrating = True
            # calib
            calibration_session.process_frame(
                queue_detection.get(), calibration_command_source.get_capture_flag(config)
            )

        elif calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0)

        if not config.local_config.has_calibration:
            print("No calibration found")
            time.sleep(0.5)

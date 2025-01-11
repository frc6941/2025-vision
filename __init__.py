import math
import multiprocessing
import pickle
import sys
import time
from multiprocessing import cpu_count
from typing import Optional

import ntcore

from calibration.CalibrationCommandSource import (
    CalibrationCommandSource,
    NTCalibrationCommandSource,
)
from calibration.CalibrationSession import CalibrationSession
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from config.config import LocalConfig, RemoteConfig, ConfigStore
from output.DetectResult import DetectResult
from output.StreamServer import MjpegServer
from output.overlay_util import *
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.Capture import DefaultCapture
# from pipeline.Capture import GStreamerCapture
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from vision_types import FiducialPoseObservation, CameraPoseObservation

DEMO_ID = 29


def process_img(
        q_time,
        q_config,
        q_result,
        q_detection,
        fps,
):
    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
    # estimator
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    tag_pose_estimator = SquareTargetPoseEstimator()
    while True:
        if not q_config.empty():
            file = open('./tmp.pkl', 'rb')
            image = pickle.load(file)
            file.close()
            p_time = q_time.get()
            p_config = q_config.get()
            time1 = time.time()
            image_observations = fiducial_detector.detect_fiducials(image, p_config)
            print("2 " + str(time.time() - time1))
            [overlay_image_observation(image, x) for x in image_observations]
            camera_pose_observation = camera_pose_estimator.solve_camera_pose(
                [x for x in image_observations if x.tag_id != DEMO_ID], p_config
            )
            demo_image_observations = [
                x for x in image_observations if x.tag_id == DEMO_ID
            ]
            demo_pose_observation: Optional[FiducialPoseObservation] = None
            if len(demo_image_observations) > 0:
                demo_pose_observation = tag_pose_estimator.solve_fiducial_pose(
                    demo_image_observations[0], p_config
                )
            send(
                q_detection=q_detection,
                timestamp=p_time,
                observation=camera_pose_observation,
                demo_observation=demo_pose_observation,
                fps=fps.value,
            )
            q_result.put(image)
            print("all " + str(time.time() - time1))
            fps.value += 1


def streaming(q_result):
    config_store = ConfigStore(LocalConfig(), RemoteConfig())
    # start stream server
    stream_server = MjpegServer()
    stream_server.start(config_store)
    # show 1 frame every display_freq frames
    # TODO: remote config
    cnt = 0
    display_freq = 5
    while True:
        if not q_result.empty():
            cnt += 1
            if cnt % display_freq == 0:
                stream_server.set_frame(q_result.get())
                cnt = 0
            else:
                q_result.get()


def send(
        q_detection,
        timestamp: float,
        observation: Optional[CameraPoseObservation],
        demo_observation: Optional[FiducialPoseObservation],
        fps: Optional[int] = None,
) -> None:
    # print(time.time())
    observation_data: list[float] = [0]
    demo_observation_data: list[float] = []
    if observation is not None:
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
        if observation.error_1 is not None and observation.pose_1 is not None:
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
    if demo_observation is not None:
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
    q_detection.put(DetectResult(fps, observation_data, demo_observation_data, math.floor(timestamp * 1000000)))


def publish_img(q_time, q_config):
    capture = DefaultCapture()
    # capture = GStreamerCapture()
    config_store = ConfigStore(LocalConfig(), RemoteConfig())
    config_source_remote: ConfigSource = NTConfigSource()
    config_source_local: ConfigSource = FileConfigSource()
    config_source_local.update(config_store)
    ntcore.NetworkTableInstance.getDefault().setServer(config_store.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config_store.local_config.device_id)
    while True:
        # update config
        config_source_remote.update(config_store)

        # get image
        success, image = capture.get_frame(config_store)
        while not success:
            print("Failed to get image")
            time.sleep(0.5)
            success, image = capture.get_frame(config_store)

        if q_config.empty():
            file = open('./tmp.pkl', 'wb')
            # noinspection PyTypeChecker
            pickle.dump(image, file)
            file.close()
            q_time.put(time.time())
            q_config.put(config_store)


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
            func=process_img,
            args=(
                queue_time,
                queue_config,
                queue_result,
                queue_detection,
                fps_count,
            ),
        )
    pool2.apply_async(func=streaming, args=(queue_result,), )
    pool3.apply_async(func=publish_img, args=(queue_time, queue_config,), )

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

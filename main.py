import cProfile
import math
import multiprocessing
import pstats
import signal
import sys
import threading
import time
from multiprocessing import cpu_count
from multiprocessing.managers import SharedMemoryManager
from multiprocessing.shared_memory import SharedMemory
from typing import Optional

import ntcore
from loguru import logger
from rich.live import Live
from rich.logging import RichHandler
from rich.table import Table

from calibration.CalibrationCommandSource import (
    CalibrationCommandSource,
    NTCalibrationCommandSource,
)
from calibration.CalibrationSession import CalibrationSession
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from config.config import ConfigStore
from output.DetectResult import DetectResult
from output.StreamServer import MjpegServer
from output.overlay_util import *
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.Capture import DefaultCapture
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from vision_types import FiducialPoseObservation, CameraPoseObservation

DEMO_ID = 29

logger.configure(handlers=[{
    "sink": RichHandler(),
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
}])


def print_profile(pr: cProfile.Profile):
    pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE).print_stats()


@logger.catch
def process_img(
        number: int,
        e_ready: threading.Event,
        m_time: SharedMemory,
        s_config: multiprocessing.Value,
        e_config: threading.Event,
        q_detection: multiprocessing.Queue,
        s_fps: dict,
        m_pic: SharedMemory,
        e_nt: threading.Event,
        m_result_sm: SharedMemory
):
    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
    # estimator
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    tag_pose_estimator = SquareTargetPoseEstimator()
    cfg = ConfigStore.model_validate_json(s_config.value)
    shape = (cfg.remote_config.camera_resolution_height,
             cfg.remote_config.camera_resolution_width,
             3)
    config_changed = True
    timestamp = numpy.ndarray((1,), dtype=numpy.float64, buffer=m_time.buf)
    # DO NOT COMBINE image2 and image !!!
    image2 = numpy.ndarray(shape, dtype=numpy.uint8, buffer=m_pic.buf)
    image_result: cv2.Mat = numpy.ndarray(shape, dtype=numpy.uint8, buffer=m_result_sm.buf)
    logger.success(f"id {number} initialized. waiting for ready signal")
    e_ready.wait()
    while True:
        try:
            image: cv2.Mat = numpy.copy(image2)
            if config_changed != e_config.is_set():
                cfg = ConfigStore.model_validate_json(s_config.value)
                logger.info("synced config")
                config_changed = e_config.is_set()
            image_observations = fiducial_detector.detect_fiducials(image, cfg)
            [overlay_image_observation(image, x) for x in image_observations]
            camera_pose_observation = camera_pose_estimator.solve_camera_pose(
                [x for x in image_observations if x.tag_id != DEMO_ID], cfg
            )
            demo_image_observations = [
                x for x in image_observations if x.tag_id == DEMO_ID
            ]
            demo_pose_observation: Optional[FiducialPoseObservation] = None
            if len(demo_image_observations) > 0:
                demo_pose_observation = tag_pose_estimator.solve_fiducial_pose(
                    demo_image_observations[0], cfg
                )
            while True:
                if not e_nt.is_set():
                    e_nt.set()
                    send(
                        q_detection=q_detection,
                        timestamp=float(timestamp[0]),
                        observation=camera_pose_observation,
                        demo_observation=demo_pose_observation,
                        fps=sum(s_fps.values())
                    )
                    e_nt.clear()
                    break;
            s_fps[number] += 1
            image_result[:] = image[:]
        except Exception as e:
            logger.exception(e)


@logger.catch
def streaming(e_ready: threading.Event, m_pic: SharedMemory, s_config: multiprocessing.Value):
    cfg = ConfigStore.model_validate_json(s_config.value)
    shape = (cfg.remote_config.camera_resolution_height,
             cfg.remote_config.camera_resolution_width,
             3)
    # start stream server
    stream_server = MjpegServer()
    stream_server.start(cfg)
    # show 1 frame every display_freq frames
    # TODO: remote config
    cnt = 0
    display_freq = 5
    image: cv2.Mat = numpy.ndarray(shape, dtype=numpy.uint8, buffer=m_pic.buf)
    logger.success("initialized. waiting for ready signal")
    e_ready.wait()
    while True:
        try:
            cnt += 1
            if cnt % display_freq == 0:
                stream_server.set_frame(image)
                cnt = 0
        except Exception as e:
            logger.exception(e)


def send(
        q_detection,
        timestamp: float,
        observation: Optional[CameraPoseObservation],
        demo_observation: Optional[FiducialPoseObservation],
        fps: Optional[int] = None,
) -> None:
    observation_data: list[float] = [0]
    demo_observation_data: list[float] = []
    if observation is not None:
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


@logger.catch
def publish_img(
        e_ready: threading.Event,
        m_time: SharedMemory,
        s_config: multiprocessing.Value,
        e_config: threading.Event,
        m_pic: SharedMemory):
    capture = DefaultCapture()
    cfg = ConfigStore.model_validate_json(s_config.value)
    shape = (cfg.remote_config.camera_resolution_height,
             cfg.remote_config.camera_resolution_width,
             3)
    timestamp = numpy.ndarray((1,), dtype=numpy.float64, buffer=m_time.buf)
    img_arr: cv2.Mat = numpy.ndarray(shape, dtype=numpy.uint8, buffer=m_pic.buf)
    config_changed = True
    logger.success("initialized. waiting for camera to be ready")
    # get one frame until success
    success, image = capture.get_frame(cfg)
    while not success:
        logger.error("failed to get image")
        time.sleep(0.5)
        success, image = capture.get_frame(cfg)

    logger.success("camera is ready")
    e_ready.set()
    while True:
        try:
            if config_changed != e_config.is_set():
                cfg = ConfigStore.model_validate_json(s_config.value)
                logger.info("synced config")
                config_changed = e_config.is_set()
            # get image
            success, image = capture.get_frame(cfg)
            while not success:
                logger.error("failed to get image")
                time.sleep(0.5)
                success, image = capture.get_frame(cfg)

            img_arr[:] = image[:]
            timestamp[0] = time.time()
        except Exception as e:
            logger.exception(e)


def debug_table(fps: dict) -> Table:
    table = Table()
    table.add_column("Process")
    table.add_column("FPS")

    for key, value in fps.items():
        table.add_row(f"process_img {key}", f"{value}")
    return table


def exit_handler(signum, frame):
    logger.debug(f"received {signal.strsignal(signum)}, exiting")
    logger.trace(f"at {frame}")
    # restore cursor
    print("\x1b[?25h")
    sys.exit(0)


signal.signal(signal.SIGINT, exit_handler)
signal.signal(signal.SIGTERM, exit_handler)

if __name__ == "__main__":
    # initialize shared memory
    smm = SharedMemoryManager()
    smm.start()

    # multiprocessing to speed up
    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool(processes=cpu_count() - 3)
    pool2 = multiprocessing.Pool(processes=1)
    pool3 = multiprocessing.Pool(processes=1)

    # variables sharing between processes
    # TODO: Remove queue_detection?
    queue_detection = manager.Queue()

    # config initialization
    config = ConfigStore()
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()

    # start NT4
    local_config_source.update(config)
    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)

    prev_config = config.model_dump_json()

    # shared memory between processes
    shape = (config.remote_config.camera_resolution_width *
             config.remote_config.camera_resolution_height * 3)
    pic_sm = smm.SharedMemory(shape)
    pic_result_sm = smm.SharedMemory(shape)
    time_sm = smm.SharedMemory(8)
    img: cv2.Mat = numpy.ndarray(shape, dtype=numpy.uint8, buffer=pic_sm.buf)

    # shared state between processes
    config_ss = manager.Value("W", "")
    config_ss.value = config.model_dump_json()
    config_se = manager.Event()
    fps_ss = manager.dict()

    ready_se = manager.Event()
    nt_se = manager.Event()

    # create cpu_count() process
    for i in range(cpu_count() - 3):  # TODO: change range when commit
        pool.apply_async(
            func=process_img,
            args=(
                i,
                ready_se,
                time_sm,
                config_ss,
                config_se,
                queue_detection,
                fps_ss,
                pic_sm,
                nt_se,
                pic_result_sm
            ),
        )
        fps_ss[i] = 0
    pool2.apply_async(func=streaming, args=(ready_se, pic_result_sm, config_ss), )
    pool3.apply_async(func=publish_img, args=(ready_se, time_sm, config_ss, config_se, pic_sm), )

    # calibration setting
    calibration_session = CalibrationSession()
    calibrating = False

    # fps_counting
    last_print = 0

    # NT
    _demo_observations_pub: ntcore.DoubleArrayPublisher
    _observations_pub: ntcore.DoubleArrayPublisher
    _fps_pub: ntcore.IntegerPublisher
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

    logger.success("initialized. waiting for ready signal")
    ready_se.wait()
    live = Live(debug_table(fps_ss), refresh_per_second=4, auto_refresh=False)
    live.start()
    while True:
        # update config
        remote_config_source.update(config)

        if prev_config != config.model_dump_json():
            prev_config = config.model_dump_json()
            config_ss.value = prev_config
            if config_se.is_set():
                config_se.clear()
            else:
                config_se.set()

        # print fps
        if time.time() - last_print > 1:
            last_print = time.time()
            live.update(debug_table(fps_ss), refresh=True)
            for i in fps_ss:
                fps_ss[i] = 0

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
                img.reshape(shape), calibration_command_source.get_capture_flag(config)
            )

        elif calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0)

        if not config.local_config.has_calibration:
            logger.warning("No calibration found")
            time.sleep(0.5)

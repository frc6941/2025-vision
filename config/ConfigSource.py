import json
from typing import Optional

import cv2
import ntcore
import numpy

from config.config import ConfigStore, RemoteConfig


class ConfigSource:
    def update(self, config_store: ConfigStore) -> None:
        raise NotImplementedError


class FileConfigSource(ConfigSource):
    CONFIG_FILENAME = "config.json"
    CALIBRATION_FILENAME = "calibration.json"

    def update(self, config_store: ConfigStore) -> None:
        # Get config
        with open(self.CONFIG_FILENAME, "r") as config_file:
            config_data = json.loads(config_file.read())
            config_store.local_config.device_id = config_data["device_id"]
            config_store.local_config.server_ip = config_data["server_ip"]
            config_store.local_config.stream_port = config_data["stream_port"]

        # Get calibration
        calibration_store = cv2.FileStorage(self.CALIBRATION_FILENAME, cv2.FILE_STORAGE_READ)
        camera_matrix = calibration_store.getNode("camera_matrix").mat()
        distortion_coefficients = calibration_store.getNode("distortion_coefficients").mat()
        calibration_store.release()
        if type(camera_matrix) == numpy.ndarray and type(distortion_coefficients) == numpy.ndarray:
            config_store.local_config.camera_matrix = camera_matrix
            config_store.local_config.distortion_coefficients = distortion_coefficients
            config_store.local_config.has_calibration = True


class NTConfigSource(ConfigSource):
    def __init__(self):
        self._init_complete: bool = False
        self._camera_id_sub: Optional[ntcore.StringSubscriber] = None
        self._camera_resolution_width_sub: Optional[ntcore.IntegerSubscriber] = None
        self._camera_resolution_height_sub: Optional[ntcore.IntegerSubscriber] = None
        self._camera_auto_exposure_sub: Optional[ntcore.DoubleSubscriber] = None
        self._camera_exposure_sub: Optional[ntcore.IntegerSubscriber] = None
        self._camera_gain_sub: Optional[ntcore.IntegerSubscriber] = None
        self._fiducial_size_m_sub: Optional[ntcore.DoubleSubscriber] = None
        self._fps: Optional[ntcore.DoubleSubscriber] = None
        self._brightness: Optional[ntcore.DoubleSubscriber] = None
        self._contrast: Optional[ntcore.DoubleSubscriber] = None
        self._buffersize: Optional[ntcore.DoubleSubscriber] = None
        self._tag_layout_sub: Optional[ntcore.StringSubscriber] = None

    def update(self, config_store: ConfigStore) -> None:
        # Initialize subscribers on first call
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + config_store.local_config.device_id + "/config")
            remote_config = RemoteConfig()
            self._camera_id_sub = nt_table.getStringTopic("camera_id").subscribe(remote_config.camera_id)
            self._camera_resolution_width_sub = nt_table.getIntegerTopic(
                "camera_resolution_width").subscribe(remote_config.camera_resolution_width)
            self._camera_resolution_height_sub = nt_table.getIntegerTopic(
                "camera_resolution_height").subscribe(remote_config.camera_resolution_height)
            self._camera_auto_exposure_sub = nt_table.getDoubleTopic(
                "camera_auto_exposure").subscribe(remote_config.camera_auto_exposure)
            self._camera_exposure_sub = nt_table.getIntegerTopic(
                "camera_exposure").subscribe(remote_config.camera_exposure)
            self._camera_gain_sub = nt_table.getIntegerTopic(
                "camera_gain").subscribe(remote_config.camera_gain)
            self._fiducial_size_m_sub = nt_table.getDoubleTopic(
                "fiducial_size_m").subscribe(remote_config.fiducial_size_m)
            self._fps = nt_table.getDoubleTopic(
                "fps").subscribe(remote_config.fps)
            self._brightness = nt_table.getDoubleTopic(
                "brightness").subscribe(remote_config.brightness)
            self._contrast = nt_table.getDoubleTopic(
                "contrast").subscribe(remote_config.contrast)
            self._buffersize = nt_table.getDoubleTopic(
                "buffersize").subscribe(remote_config.buffersize)
            self._init_complete = True

        # Read config data
        # FIXME: Move some out of NT
        config_store.remote_config.camera_id = self._camera_id_sub.get()
        config_store.remote_config.camera_resolution_width = self._camera_resolution_width_sub.get()
        config_store.remote_config.camera_resolution_height = self._camera_resolution_height_sub.get()
        config_store.remote_config.camera_auto_exposure = self._camera_auto_exposure_sub.get()
        config_store.remote_config.camera_exposure = self._camera_exposure_sub.get()
        config_store.remote_config.camera_gain = self._camera_gain_sub.get()
        config_store.remote_config.fiducial_size_m = self._fiducial_size_m_sub.get()
        # FIXME: Type cast from double to int
        config_store.remote_config.fps = int(self._fps.get())
        config_store.remote_config.brightness = int(self._brightness.get())
        config_store.remote_config.contrast = int(self._contrast.get())
        config_store.remote_config.buffersize = int(self._buffersize.get())

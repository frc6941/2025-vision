from dataclasses import dataclass

import numpy.typing


@dataclass
class LocalConfig:
    # edit it in config.json
    device_id: str = "northstar_0"
    server_ip: str = "10.96.20.2"
    stream_port: int = 8000
    has_calibration: bool = True  # TODO: calib
    camera_matrix: numpy.typing.NDArray[numpy.float64] = numpy.array([])
    distortion_coefficients: numpy.typing.NDArray[numpy.float64] = numpy.array([])


@dataclass
class RemoteConfig:
    camera_id: str = "2"
    camera_resolution_width: int = 1280
    camera_resolution_height: int = 720
    camera_auto_exposure: float = 0.25
    camera_exposure: int = -10
    camera_gain: int = 1
    fiducial_size_m: float = 0.1675
    tag_layout: any = None
    fps: int = 60
    brightness: int = 35
    contrast: int = 60
    buffersize: int = 1


@dataclass
class ConfigStore:
    local_config: LocalConfig
    remote_config: RemoteConfig

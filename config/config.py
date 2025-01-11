from dataclasses import dataclass, field

import numpy.typing


@dataclass
class LocalConfig:
    # edit it in config.json
    device_id: str = "northstar_0"
    server_ip: str = "10.96.20.2"
    stream_port: int = 8000
    has_calibration: bool = True
    camera_matrix: numpy.typing.NDArray[numpy.float64] = field(default_factory=lambda: numpy.array([]))
    distortion_coefficients: numpy.typing.NDArray[numpy.float64] = field(default_factory=lambda: numpy.array([]))


@dataclass
class RemoteConfig:
    # camera_id: str = "/dev/v4l/by-path/platform-fc880000.usb-usb-0:1:1.0-video-index0"
    camera_id: str = "/dev/video_cam1"
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

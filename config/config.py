import base64
import io
from typing import TypeAlias, Annotated, Optional

import numpy.typing
from pydantic import BaseModel, PlainSerializer, ConfigDict, PlainValidator


# efforts to make ndarray serializable...
def nd_array_custom_validator(x: str) -> numpy.ndarray:
    f = io.BytesIO(base64.b64decode(x))
    try:
        return numpy.load(f, encoding="bytes")
    finally:
        f.close()


def nd_array_custom_serializer(x) -> str:
    f = io.BytesIO()
    try:
        numpy.save(f, x)
        return base64.b64encode(f.getvalue()).decode()
    finally:
        f.close()


PydanticNDArray: TypeAlias = Annotated[
    numpy.typing.NDArray[numpy.float64],
    PlainValidator(nd_array_custom_validator),
    PlainSerializer(nd_array_custom_serializer, return_type=str)
]


class LocalConfig(BaseModel):
    # edit it in config.json
    device_id: str = "northstar_0"
    server_ip: str = "10.96.20.2"
    stream_port: int = 8000
    has_calibration: bool = True
    camera_matrix: PydanticNDArray = numpy.array([])
    distortion_coefficients: PydanticNDArray = numpy.array([])

    model_config = ConfigDict(arbitrary_types_allowed=True)


class RemoteConfig(BaseModel):
    # camera_id: str = "/dev/v4l/by-path/platform-fc880000.usb-usb-0:1:1.0-video-index0"
    # camera_id: str = "/dev/video0"
    # camera_id: str = "0"
    camera_id: str = "/dev/video_cam1"
    camera_resolution_width: int = 1280
    camera_resolution_height: int = 720
    camera_auto_exposure: float = 0.25
    camera_exposure: int = -10
    camera_gain: int = 1
    fiducial_size_m: float = 0.1675
    tag_layout: Optional[dict] = None
    fps: int = 60
    brightness: int = 35
    contrast: int = 60
    buffersize: int = 1


class ConfigStore(BaseModel):
    local_config: LocalConfig = LocalConfig()
    remote_config: RemoteConfig = RemoteConfig()

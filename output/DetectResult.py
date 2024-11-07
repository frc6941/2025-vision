from typing import Union

from output.overlay_util import *
from vision_types import CameraPoseObservation, FiducialPoseObservation


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
        print("created")

    # def __init__(self):
    #     self.config: ConfigStore
    #     self.time: float
    #     self.observation: Union[CameraPoseObservation, None]
    #     self.demo_observation: Union[FiducialPoseObservation, None]
    #     self.fps_count: int

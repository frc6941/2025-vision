import math
from typing import Optional

import ntcore

from config.config import ConfigStore
from vision_types import CameraPoseObservation, FiducialPoseObservation


class OutputPublisher:
    def send(
            self,
            config_store: ConfigStore,
            timestamp: float,
            observation: Optional[CameraPoseObservation],
            demo_observation: Optional[FiducialPoseObservation],
            fps: Optional[int] = None,
    ) -> None:
        raise NotImplementedError


class NTOutputPublisher(OutputPublisher):
    def __init__(self):
        self._init_complete: bool = False
        self._observations_pub: Optional[ntcore.DoubleArrayPublisher] = None
        self._observations_pub: Optional[ntcore.DoubleArrayPublisher] = None
        self._fps_pub: Optional[ntcore.IntegerPublisher] = None
        self._demo_observations_pub: Optional[ntcore.DoubleArrayPublisher] = None

    def send(
            self,
            config_store: ConfigStore,
            timestamp: float,
            observation: Optional[CameraPoseObservation],
            demo_observation: Optional[FiducialPoseObservation],
            fps: Optional[int] = None,
    ) -> None:
        # Initialize publishers on first call
        if not self._init_complete:
            ntcore.NetworkTableInstance.getDefault().setServer(config_store.local_config.server_ip)
            ntcore.NetworkTableInstance.getDefault().startClient4(config_store.local_config.device_id)
            self._init_complete = True
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + config_store.local_config.device_id + "/output"
            )
            self._observations_pub = nt_table.getDoubleArrayTopic(
                "observations"
            ).publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True)
            )
            self._demo_observations_pub = nt_table.getDoubleArrayTopic(
                "demo_observations"
            ).publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True)
            )
            self._fps_pub = nt_table.getIntegerTopic("fps").publish()

        # Send data
        if fps is not None:
            self._fps_pub.set(fps)
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
        self._observations_pub.set(observation_data, math.floor(timestamp * 1000000))
        self._demo_observations_pub.set(
            demo_observation_data, math.floor(timestamp * 1000000)
        )

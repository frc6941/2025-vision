from typing import Optional

import cv2
import numpy
from wpimath.geometry import *

from config.config import ConfigStore
from pipeline.coordinate_systems import (opencv_pose_to_wpilib,
                                         wpilib_translation_to_opencv)
from vision_types import CameraPoseObservation, FiducialImageObservation


class CameraPoseEstimator:
    def __init__(self) -> None:
        pass

    def solve_camera_pose(self, image_observations: list[FiducialImageObservation], config_store: ConfigStore) -> \
            Optional[
                CameraPoseObservation]:
        raise NotImplementedError


class MultiTargetCameraPoseEstimator(CameraPoseEstimator):
    def __init__(self) -> None:
        super().__init__()

    def solve_camera_pose(self, image_observations: list[FiducialImageObservation], config_store: ConfigStore) -> \
            Optional[
                CameraPoseObservation]:
        # Exit if no tag layout available
        if config_store.local_config.tag_layout is None:
            return None

        # Exit if no observations available
        if len(image_observations) == 0:
            return None

        # Create set of object and image points
        fid_size = config_store.remote_config.fiducial_size_m
        object_points = []
        image_points = []
        tag_ids = []
        tag_poses = []
        for observation in image_observations:
            tag_pose = None
            for tag_data in config_store.local_config.tag_layout["tags"]:
                if tag_data["ID"] == observation.tag_id:
                    tag_pose = Pose3d(
                        Translation3d(
                            tag_data["pose"]["translation"]["x"],
                            tag_data["pose"]["translation"]["y"],
                            tag_data["pose"]["translation"]["z"]
                        ),
                        Rotation3d(Quaternion(
                            tag_data["pose"]["rotation"]["quaternion"]["W"],
                            tag_data["pose"]["rotation"]["quaternion"]["X"],
                            tag_data["pose"]["rotation"]["quaternion"]["Y"],
                            tag_data["pose"]["rotation"]["quaternion"]["Z"]
                        )))
            if tag_pose is not None:
                # Add object points by transforming from the tag center
                corner_0 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_1 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_2 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, fid_size / 2.0), Rotation3d())
                corner_3 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, fid_size / 2.0), Rotation3d())
                object_points += [
                    wpilib_translation_to_opencv(corner_0.translation()),
                    wpilib_translation_to_opencv(corner_1.translation()),
                    wpilib_translation_to_opencv(corner_2.translation()),
                    wpilib_translation_to_opencv(corner_3.translation())
                ]

                # Add image points from observation
                image_points += [
                    [observation.corners[0][0][0], observation.corners[0][0][1]],
                    [observation.corners[0][1][0], observation.corners[0][1][1]],
                    [observation.corners[0][2][0], observation.corners[0][2][1]],
                    [observation.corners[0][3][0], observation.corners[0][3][1]]
                ]

                # Add tag ID and pose
                tag_ids.append(observation.tag_id)
                tag_poses.append(tag_pose)

        # Single tag, return two poses
        if len(tag_ids) == 1:
            object_points = numpy.array([[-fid_size / 2.0, fid_size / 2.0, 0.0],
                                         [fid_size / 2.0, fid_size / 2.0, 0.0],
                                         [fid_size / 2.0, -fid_size / 2.0, 0.0],
                                         [-fid_size / 2.0, -fid_size / 2.0, 0.0]])
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(object_points, numpy.array(image_points),
                                                              config_store.local_config.camera_matrix,
                                                              config_store.local_config.distortion_coefficients,
                                                              flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except cv2.error:
                return None

            # Calculate WPILib camera poses
            field_to_tag_pose = tag_poses[0]
            camera_to_tag_pose_0 = opencv_pose_to_wpilib(tvecs[0], rvecs[0])
            camera_to_tag_pose_1 = opencv_pose_to_wpilib(tvecs[1], rvecs[1])
            camera_to_tag_0 = Transform3d(camera_to_tag_pose_0.translation(), camera_to_tag_pose_0.rotation())
            camera_to_tag_1 = Transform3d(camera_to_tag_pose_1.translation(), camera_to_tag_pose_1.rotation())
            field_to_camera_0 = field_to_tag_pose.transformBy(camera_to_tag_0.inverse())
            field_to_camera_1 = field_to_tag_pose.transformBy(camera_to_tag_1.inverse())
            field_to_camera_pose_0 = Pose3d(field_to_camera_0.translation(), field_to_camera_0.rotation())
            field_to_camera_pose_1 = Pose3d(field_to_camera_1.translation(), field_to_camera_1.rotation())

            # Return result
            return CameraPoseObservation(tag_ids, field_to_camera_pose_0, float(errors[0][0]), field_to_camera_pose_1,
                                         float(errors[1][0]))

        # Multi-tag, return one pose
        else:
            # Run SolvePNP with all tags
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(numpy.array(object_points), numpy.array(image_points),
                                                              config_store.local_config.camera_matrix,
                                                              config_store.local_config.distortion_coefficients,
                                                              flags=cv2.SOLVEPNP_SQPNP)
            except cv2.error:
                return None

            # Calculate WPILib camera pose
            camera_to_field_pose = opencv_pose_to_wpilib(tvecs[0], rvecs[0])
            camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
            field_to_camera = camera_to_field.inverse()
            field_to_camera_pose = Pose3d(field_to_camera.translation(), field_to_camera.rotation())

            # Return result
            return CameraPoseObservation(tag_ids, field_to_camera_pose, float(errors[0][0]), None, None)

import math

import numpy
import numpy.typing
from wpimath.geometry import *


def opencv_pose_to_wpilib(tvec: numpy.typing.NDArray[numpy.float64],
                          rvec: numpy.typing.NDArray[numpy.float64]) -> Pose3d:
    # noinspection PyTypeChecker
    return Pose3d(
        Translation3d(float(tvec[2][0]), float(-tvec[0][0]), float(-tvec[1][0])),
        Rotation3d(
            numpy.array([float(rvec[2][0]), float(-rvec[0][0]), float(-rvec[1][0])]),
            math.sqrt(math.pow(float(rvec[0][0]), 2) + math.pow(float(rvec[1][0]), 2) + math.pow(float(rvec[2][0]), 2))
        ))


def wpilib_translation_to_opencv(translation: Translation3d) -> list[float]:
    return [-translation.Y(), -translation.Z(), translation.X()]

class Config:
    class NetworkConfig:
        CameraID: str = "200"  # USB ID to connect the camera
        DeviceName: str = "OrangePi5Plus1"
        ServerIP: str = "10.96.20.2"

    class ConstantConfig:
        FiducialSize: float = 0.1675  # meter

    class RemoteConfig:
        class Camera:
            CameraResolutionWidth: int = 1280
            CameraResolutionHeight: int = 720
            CameraAutoExposure: int = 1
            CameraExposure: int = 60
            CameraGain: int = 1

        class Calibration:
            OnCalibration = False

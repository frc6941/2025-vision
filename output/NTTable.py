import ntcore

from Config.Config import Config


class NTTable:
    def init(self):
        ntcore.NetworkTableInstance.getDefault().setServer(Config.NetworkConfig.ServerIP)
        ntcore.NetworkTableInstance.getDefault().startClient4(Config.NetworkConfig.DeviceName)

    nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
        "/" + Config.NetworkConfig.DeviceName + "/output")

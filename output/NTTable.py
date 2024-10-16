import time

import ntcore

from Config.Config import Config


class NTTable:
    i = 0
    ntcore.NetworkTableInstance.getDefault().setServer(Config.NetworkConfig.ServerIP)
    ntcore.NetworkTableInstance.getDefault().startClient4(Config.NetworkConfig.DeviceName)
    timeStampPublisher: ntcore.FloatPublisher
    nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
        "/" + Config.NetworkConfig.DeviceName + "/output")
    timeStampPublisher = nt_table.getFloatTopic("timestamp").publish()
    timeStampPublisher2 = nt_table.getFloatTopic("timestamp2").publish()

    def periodic(self):
        self.timeStampPublisher.set(time.time())
        self.timeStampPublisher2.set(self.i)
        self.i += 1

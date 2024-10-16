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
    timeStampPublisher = nt_table.getFloatTopic("timestamp").publish(
        ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True))
    timeStampPublisher2 = nt_table.getFloatTopic("timestamp2").publish()
    _instance = None

    def __init__(self):
        pass

    def periodic(self):
        self.timeStampPublisher.set(time.time())
        self.timeStampPublisher2.set(self.i)
        self.i += 1
        time.sleep(1)
        self.timeStampPublisher2.set(self.i)

    @classmethod
    def getInstance(cls):
        if not cls._instance:
            cls._instance = NTTable()
        return cls._instance

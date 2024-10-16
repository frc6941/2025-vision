from output.NTTable import NTTable


def init():
    pass


def periodic():
    NTTable().getInstance().periodic()


init()
while True:
    periodic()

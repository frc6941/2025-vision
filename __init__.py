from output.NTTable import NTTable


def init():
    NTTable.init()


def periodic():
    NTTable.nt_table.putNumber("111", 11)


while True:
    periodic()

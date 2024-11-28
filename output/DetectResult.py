class DetectResult:
    def __init__(self,
                 fps: int,
                 observation: list[float],
                 demo_observation: list[float],
                 time: int):
        self.fps: int = fps
        self.observation: list[float] = observation
        self.demo_observation: list[float] = demo_observation
        self.time: int = time

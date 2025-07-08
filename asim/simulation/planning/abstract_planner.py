# TODO: Remove or implement this placeholder


class AbstractPlanner:
    def __init__(self):
        self._arg = None

    def step(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

"""Simulated backends for the intern example, not real DWPose inference."""

from copy import deepcopy


class ExampleDetector:
    def __init__(self, outputs):
        self.outputs = deepcopy(outputs)
        self.calls = 0

    def detect(self, image):
        self.calls += 1
        return deepcopy(self.outputs)


class ExampleEstimator:
    def __init__(self, outputs):
        self.outputs = deepcopy(outputs)
        self.calls = 0

    def estimate(self, image, boxes):
        self.calls += 1
        if len(boxes) != len(self.outputs):
            raise AssertionError('Estimator needs one output for each input box')
        return deepcopy(self.outputs)

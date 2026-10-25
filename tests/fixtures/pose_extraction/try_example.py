"""Run supplied examples against the wrapper after the intern implements it."""

import json
import math
from pathlib import Path
import sys

import numpy as np

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR.parents[2]))

from example_backends import ExampleDetector, ExampleEstimator


def main():
    try:
        from src.pose_models.extraction.dwpose import (
            DWPoseFrameExtractor,
            PoseExtractionError,
        )
    except ModuleNotFoundError as error:
        raise SystemExit(
            'The wrapper is not implemented yet. Add '
            'src/pose_models/extraction/dwpose.py, then run this command again.'
        ) from error

    for path in sorted(EXAMPLE_DIR.glob('*.json')):
        data = json.loads(path.read_text(encoding='utf-8'))
        spec = data['image']
        image = np.full(
            (spec['height'], spec['width'], spec['channels']),
            spec['fill'], dtype=np.uint8,
        )
        unchanged = image.copy()
        detector = ExampleDetector(data['detector_outputs'])
        estimator = ExampleEstimator(data['estimator_outputs'])
        wrapper = DWPoseFrameExtractor(
            detector=detector, estimator=estimator, model_info=data['model_info']
        )
        try:
            result = wrapper.extract_frame(
                image, reviewed_boxes=data['reviewed_boxes']
            )
        except PoseExtractionError:
            if data.get('expected_error') != 'PoseExtractionError':
                raise
        else:
            if 'expected_error' in data:
                raise AssertionError(f'{path.name}: expected PoseExtractionError')
            result = dict(result)
            timing = result.pop('timing_ms')
            if set(timing) != {'detector', 'estimator', 'total'}:
                raise AssertionError(f'{path.name}: unexpected timing fields')
            for value in timing.values():
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    raise AssertionError(f'{path.name}: invalid timing')
            if detector.calls == 0 and timing['detector'] != 0:
                raise AssertionError(f'{path.name}: skipped detector timing must be zero')
            if estimator.calls == 0 and timing['estimator'] != 0:
                raise AssertionError(f'{path.name}: skipped estimator timing must be zero')
            if result != data['expected']:
                raise AssertionError(f'{path.name}: output differs from supplied expected result')
            json.dumps(result, allow_nan=False)
        if not np.array_equal(image, unchanged):
            raise AssertionError(f'{path.name}: image was changed')
        calls = {'detector': detector.calls, 'estimator': estimator.calls}
        if calls != data['expected_calls']:
            raise AssertionError(f'{path.name}: wrong backend call counts {calls}')
        print(f'PASS {path.name}')


if __name__ == '__main__':
    main()

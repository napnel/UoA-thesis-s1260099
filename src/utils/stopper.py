from collections import defaultdict, deque
import numpy as np
from ray.tune.stopper import Stopper


class MedianStopper(Stopper):
    def __init__(self, metric, num_results, grace_period):
        self._metric = metric
        self._num_results = num_results
        self._grace_period = grace_period

        self._iter = defaultdict(lambda: 0)
        self._trial_results = defaultdict(lambda: deque(maxlen=self._num_results))

    def __call__(self, trial_id, result):
        metric_result = result.get(self._metric)
        self._trial_results[trial_id].append(metric_result)
        self._iter[trial_id] += 1

        if self._iter[trial_id] < self._grace_period:
            return False

        if len(self._trial_results[trial_id]) < self._num_results:
            return False

        try:
            current_median = np.median(self._trial_results[trial_id])
        except Exception:
            current_median = -np.inf

        return metric_result < current_median

    def stop_all(self):
        return False

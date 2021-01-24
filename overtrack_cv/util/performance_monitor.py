import logging
import time
from collections import defaultdict, deque
from typing import Any, DefaultDict, Deque, List, Optional, Tuple

import numpy as np
import tabulate

from overtrack_cv.frame import Timings


class PerformanceMonitor:

    logger = logging.getLogger("PerformanceMonitor")

    def __init__(self, fps: float, report_frequency: float = 60, minutes: int = 5):
        self.perf: Deque[Tuple[float, Timings, Optional[int]]] = deque(maxlen=int(fps * 60 * minutes))
        self.report_frequency = report_frequency
        self.last_shown: float = time.time()

    def submit(self, t: Timings, qsize: Optional[int] = None) -> None:
        self.perf.append((time.time(), t, qsize))
        if time.time() - self.last_shown > self.report_frequency:
            self.last_shown = time.time()
            self.report()

    def report(self) -> None:
        values: DefaultDict[str, List[float]] = defaultdict(list)
        for ts, timing, qsize in self.perf:
            for k, v in timing.items():
                values[k].append(v)
            if qsize is not None:
                values["QSIZE"].append(qsize)
            values["TOTAL"].append(timing.total)

        count = len(values["TOTAL"])
        self.logger.info(
            "Got performance stats: \n"
            + tabulate.tabulate(
                [
                    [
                        k,
                        int((len(v) / count) * 100) if k != "QSIZE" else "",
                        np.median(v),
                        np.percentile(v, 10),
                        np.percentile(v, 90),
                        np.percentile(v, 99),
                    ]
                    for (k, v) in sorted(values.items(), key=self.keysort)
                ],
                headers=["PROCESSOR", "active%", "median", "10%", "90%", "99%"],
            )
        )

    FIRST = ["QSIZE", "fetch", "in_queue"]

    def keysort(self, item: Tuple[str, Any]) -> Tuple[int, str]:
        key = item[0]
        if key in self.FIRST:
            return -self.FIRST[::-1].index(key), key
        elif key != "TOTAL":
            return 1, key
        else:
            return 2, key

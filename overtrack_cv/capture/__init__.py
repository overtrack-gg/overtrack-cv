import time
from typing import Optional, Union

from overtrack_cv.frame import Frame


class Capture:
    def get(self) -> Optional[Union[Frame, Exception]]:
        raise NotImplementedError()

    def close(self):
        pass


class RateLimitingCapture(Capture):
    def __init__(self, capture: Capture, fps: int):
        self.capture = capture
        self.capture_fps = fps
        self.min_wait = 1 / fps
        self.last_capture: Optional[float] = None

    def get(self) -> Optional[Frame]:
        if self.last_capture:
            since_last = time.time() - self.last_capture
            wait = self.min_wait - since_last
            if wait > 0:
                time.sleep(wait)
        t0 = time.time()
        frame = self.capture.get()
        if isinstance(frame, Exception):
            raise frame

        if not frame:
            return None

        t1 = time.time()

        # frame.timings['Capture'] = (t1 - t0) * 1000
        frame.timestamp_delta = (t0 - self.last_capture) if self.last_capture else 0
        self.last_capture = t0
        return frame

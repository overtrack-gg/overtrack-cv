import logging
import mmap
import struct
import time
from collections import deque
from threading import Lock, Thread
from typing import Optional, Sequence, Tuple

import cv2
import numpy as np

from overtrack_cv.capture.shmem import SharedMemorySource
from overtrack_cv.frame import Frame
from overtrack_cv.games.processor import Processor
from overtrack_cv.util import s2ts
from overtrack_cv.util.logging_config import config_logger

DEBUG = False

logger = logging.getLogger("SharedMemoryCapture")
zimg = np.zeros((1080, 1920, 3), dtype=np.uint8)


def is_blank(frame: Frame) -> bool:
    return np.sum(frame.image) == 0


class SharedMemoryCapture:
    # Recheck what game this is every 5s
    RECHECK_GAME_INTERVAL = 5

    # Consider 5 minutes of blank/black frames to mean the game has exited
    BLANK_FRAMES_TIMEOUT = 5 * 60

    # Consider 10s of no frames received to mean that the source is no longer sending
    NO_FRAMES_TIMEOUT = 10

    def __init__(
        self,
        processor_checks: Sequence[Tuple[str, str, Processor]] = (),
        debug_frames=False,
        include_unscaled=False,
        update_frequency=0.5,
        queue_size: int = 3,
    ):
        self.debug_frames = debug_frames
        self.include_unscaled = include_unscaled

        self.shmem_name = "overtrack"
        self.last_index: Optional[int] = None
        self.update_frequency = update_frequency

        self.processor_checks = list(processor_checks)
        # self.processor_checks: List[Tuple[str, str, Processor]] = [
        #     ('overwatch', 'Overwatch', ShortCircuitProcessor(OverwatchMenuProcessor(), LoadingMapProcessor(), order_defined=False)),
        #     ('apex', 'Apex Legends', ApexMenuProcessor()),
        #     ('valorant', 'Valorant', ValorantHomeScreenProcessor())
        # ]

        self.active = False
        self._stop = False
        self.started = time.time()
        self.last_source: Optional[SharedMemorySource] = None
        self.current_game: Optional[str] = None
        self.last_checked_game = 0.0
        self.last_frame_received = 0.0
        self.last_nonblank_frame = 0.0

        self.queue = deque(maxlen=queue_size)
        self.lock = Lock()

        self.thread: Optional[Thread] = None
        self.started = False

    def _resize_image(self, image: np.ndarray, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        if image.shape[0] == 1080:
            return image

        scale = 1080 / image.shape[0]
        if image.shape[1] * scale > 2560:
            scale = 2560 / image.shape[1]

        cscale = min(scale, 2)
        scaled = cv2.resize(image, (0, 0), fx=cscale, fy=cscale, interpolation=interpolation)
        if scale <= 2:
            return scaled
        else:
            return cv2.copyMakeBorder(
                scaled,
                0,
                max(1080 - scaled.shape[0], 0),
                0,
                max(1920 - scaled.shape[1], 0),
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

    def fetch_next_frame(self) -> bool:
        shmem_name = None
        shmem_game = None
        for suffix, game, _ in self.processor_checks + [("", None, None)]:
            check_name = self.shmem_name
            if suffix:
                check_name += ":" + suffix
            with mmap.mmap(0, 16, check_name, mmap.ACCESS_READ) as shmem:
                w, h, l, i = struct.unpack("<IIII", shmem.read(16))
                shmem.close()

            if w and h and l:
                if not self.last_index or i < self.last_index:
                    logger.debug(
                        f"Detected restart in shmem header - index: {self.last_index}->{i} - w={w}, h={h}, l={l}"
                    )
                if i != self.last_index:
                    shmem_name = check_name
                    shmem_game = game
                    break

        if not shmem_name:
            return False

        self.last_index = i

        with mmap.mmap(0, 16 + (h * l), shmem_name, mmap.ACCESS_READ) as shmem:
            shmem.seek(16)
            image = shmem.read(h * l)
            shmem.close()

        now = time.time()

        image = np.frombuffer(image, dtype=np.uint8).reshape((h, l // 4, 4))[:, :w]
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        resize_image = self._resize_image(image)

        source = SharedMemorySource((image.shape[1], image.shape[0]), w, h, l, shmem_name)
        if source != self.last_source:
            self.last_source = source
            logger.info(f"Found new source: {source}")

        frame = Frame.create(
            image=resize_image[:, :, :3],
            timestamp=round(now, 4),
            image_alpha=resize_image[:, :, 3],
            relative_timestamp=round(now - self.started, 4),
            source_frame_no=i,
            source=self.last_source,
            debug=self.debug_frames,
        )
        if self.include_unscaled:
            frame.image_unscaled = image[:, :, :3]

        self.last_frame_received = now
        if not self.active:
            logger.warning(f"Got frame from {shmem_name!r} - setting active=True")
            self.active = True

        if not shmem_game:
            # generic shmem - did not specify game
            if now - self.last_checked_game > self.RECHECK_GAME_INTERVAL:
                self.last_checked_game = now
                blank = is_blank(frame)
                if blank:
                    logger.info(f"Rechecking mode: got blank frame")
                    if now - self.last_nonblank_frame > self.BLANK_FRAMES_TIMEOUT:
                        logger.warning(
                            f"Had {(now - self.last_nonblank_frame) / 60:.1f}m of blank frames - setting game=None"
                        )
                        self.current_game = None
                else:
                    self.last_nonblank_frame = now
                    for n, g, p in self.processor_checks:
                        if p.process(frame):
                            logger.debug(f"Rechecking mode: got {self.current_game}")
                            if self.current_game != g:
                                self.current_game = g
                                logger.warning(
                                    f"{n} matched on frame - setting game={g!r} on capture with unspecified game"
                                )
                            break
                    else:
                        # no game matched, keep in current state
                        pass
        else:
            if self.current_game != shmem_game:
                self.current_game = shmem_game
                logger.warning(f"Setting game={self.current_game!r} from shmem={shmem_game}")

        frame.source_name = "Shared Memory"
        frame.valid = self.current_game is not None
        frame.game = self.current_game
        # frame.timings['SharedMemoryCapture'] = time.time() - now

        self.queue.append(frame)
        return True

    def run(self):
        self.started = True
        last_valid = 0.0
        while not self._stop:
            try:
                valid = self.fetch_next_frame()
            except:
                logger.exception("Failed to fetch next frame")
                valid = False

            now = time.time()
            if valid:
                last_valid = now
            elif now - last_valid > self.NO_FRAMES_TIMEOUT:
                if self.active:
                    logger.warning(
                        f"Did not get frame after {now - last_valid:.0f}s, setting active=False, unsetting current_game (was {self.current_game!r})"
                    )
                    self.active = None
                    self.current_game = None

            time.sleep(self.update_frequency)

    def start(self):
        with self.lock:
            if self.thread is not None:
                raise ValueError(f"{self.__class__.__name__} is already started")
            self.thread = Thread(target=self.run, name=self.__class__.__name__, daemon=True)
            self.thread.start()

    def get(self) -> Optional[Frame]:
        with self.lock:
            if not self.started:
                raise ValueError(
                    f"Cannot get frame asynchronously from non-started {self.__class__.__name__} - use .start()/.run() first or try .get_sync()"
                )
            if len(self.queue):
                return self.queue.pop()
        return None

    def get_blocking(self) -> Optional[Frame]:
        while True:
            with self.lock:
                if not self.started:
                    raise ValueError(
                        f"Cannot get frame asynchronously from non-started {self.__class__.__name__} - use .start()/.run() first or try .get_sync()"
                    )
                if len(self.queue):
                    return self.queue.pop()
            time.sleep(0.1)

    def get_sync(self) -> Optional[Frame]:
        with self.lock:
            if self.started:
                raise ValueError(
                    f"Cannot get frame synchronously from started {self.__class__.__name__} - remove .start()/.run() or use .get()"
                )
            self.queue.clear()
            if not self.fetch_next_frame():
                # TODO: invalid checks
                return None
            assert self.qsize() == 1
            return self.queue.pop()

    def qsize(self):
        return len(self.queue)

    def __str__(self) -> str:
        return (
            f"SharedMemoryCapture("
            f"active={self.active}, "
            f"current_game={self.current_game!r}, "
            f"last_frame={s2ts(time.time() - self.last_frame_received, zpad=False)} ago, "
            f"qsize={self.qsize()}"
            f")"
        )


def main() -> None:
    config_logger("shmem_source", logging.DEBUG, False)

    cap = SharedMemoryCapture(debug_frames=True)
    cap.start()
    while True:
        frame = cap.get()
        if not frame:
            print("x")
            time.sleep(4)
            continue
        elif isinstance(frame, Exception):
            raise frame

        cv2.imshow("frame", frame.debug_image)
        print(frame.source)
        cv2.waitKey(1000)


if __name__ == "__main__":
    main()

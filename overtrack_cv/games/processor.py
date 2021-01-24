import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from overtrack_cv.frame import Frame


# TODO: Exception recording as part of processor tracing
@dataclass
class ProcessorTracing:
    name: str
    qualname: str
    duration: float
    result: bool
    subprocessors: Optional[List["ProcessorTracing"]] = None
    processor_metadata: Optional[Dict] = None


class Processor:
    def run(self, frame: Frame) -> ProcessorTracing:
        t0 = time.perf_counter()
        r = self.process(frame)
        t1 = time.perf_counter()
        return ProcessorTracing(
            name=self.__class__.__name__,
            qualname=self.__class__.__qualname__,
            duration=round(t1 - t0, 4),
            result=r,
        )

    def process(self, frame: Frame) -> bool:
        raise NotImplemented()

    def eager_load(self):
        pass

    def update(self):
        pass


class OrderedProcessor(Processor):
    """
    Run all processors in the predefined order.
    Processing evaluates True if _condition_ of the processors returns True (e.g. _condition_ might be all() or any())
    """

    def __init__(self, *processors: Processor, condition=any):
        self.processors = processors
        self.condition = condition

    def run(self, frame: Frame) -> ProcessorTracing:
        t0 = time.perf_counter()
        subprocessors = [p.run(frame) for p in self.processors]
        t1 = time.perf_counter()
        return ProcessorTracing(
            name=self.__class__.__name__,
            qualname=self.__class__.__qualname__,
            duration=round(t1 - t0, 4),
            result=self.condition([sp.result for sp in subprocessors]),
            subprocessors=subprocessors,
        )

    def eager_load(self):
        for p in self.processors:
            p.eager_load()

    def update(self):
        for p in self.processors:
            p.update()


class ConditionalProcessor(Processor):
    """
    Run the processor IFF the provided condition evaluates to True

    If this is a lookbehind ConditionalProcessor, run the processor if the condition was true for any of the frames in the lookbehind window.
    Note that condition evaluation is cached.


    """

    logger = logging.getLogger("ConditionalProcessor")

    def __init__(
        self,
        processor: Processor,
        condition: Callable[[Frame], object],  # condition may return falsey objects
        lookbehind: Optional[int] = None,
        lookbehind_behaviour: Callable[[Iterable[Any]], bool] = any,
        default_without_history: bool = True,
        log: bool = False,
    ):
        self.processor = processor
        self.condition = condition
        self.lookbehind = lookbehind
        self.lookbehind_behaviour = lookbehind_behaviour
        self.log = log
        if lookbehind:
            self.history = deque([default_without_history] * lookbehind, maxlen=lookbehind)

        self._last_metadata: Optional[Dict] = None

    def run(self, frame: Frame) -> ProcessorTracing:
        processor_metadata = {}
        if self.lookbehind:
            # check before popleft in case checking raises and exception
            val = bool(self.condition(frame))
            processor_metadata["condition_result"] = val
            self.history.popleft()
            self.history.append(val)
            result = self.lookbehind_behaviour(self.history)
            if self.log:
                self.logger.info(f"ConditionalProcessor on {self.processor}: {self.history} => {result}")
        else:
            result = bool(self.condition(frame))
            if self.log:
                self.logger.info(f"ConditionalProcessor on {self.processor}: {result}")
        processor_metadata["processor_run"] = result
        if result:
            t0 = time.perf_counter()
            subprocessors = [self.processor.run(frame)]
            r = subprocessors[0].result
            t = time.perf_counter() - t0
        else:
            subprocessors = []
            t = 0
            r = False
        return ProcessorTracing(
            name=self.__class__.__name__,
            qualname=self.__class__.__qualname__,
            duration=round(t, 4),
            result=r,
            subprocessors=subprocessors,
            processor_metadata=processor_metadata,
        )

    def eager_load(self):
        self.processor.eager_load()

    def update(self):
        self.processor.update()


class ShortCircuitProcessor(Processor):
    """
    Run the processors until one of them returns True

    If order_defined=False then the processor that returned True on the previous frame is run first
    """

    def __init__(
        self,
        *processors: Processor,
        order_defined: bool,
        invert: bool = False,
        log: bool = False,
    ):
        self.processors = list(processors)
        self.order_defined = order_defined
        self.invert = invert
        self.log = log
        if not self.order_defined:
            self.last_processor = processors[0]

        self._last_metadata: Optional[Dict] = None

    def should_bail(self, result: bool) -> bool:
        if self.invert:
            return not result
        else:
            return result

    def run(self, frame: Frame) -> ProcessorTracing:
        processor_metadata = {}
        subprocessors = []
        bailed = False
        t0 = time.perf_counter()
        for p in list(self.processors):
            subprocessors.append(p.run(frame))
            if self.should_bail(subprocessors[-1].result):
                if not self.order_defined:
                    processor_metadata["reordered"] = self.processors[0] is not p
                    self.processors.remove(p)
                    self.processors.insert(0, p)
                bailed = True
                break
        return ProcessorTracing(
            name=self.__class__.__name__,
            qualname=self.__class__.__qualname__,
            duration=round(time.perf_counter() - t0, 4),
            result=bailed,
            subprocessors=subprocessors,
            processor_metadata=processor_metadata,
        )

    def eager_load(self):
        for p in self.processors:
            p.eager_load()

    def update(self):
        for p in self.processors:
            p.eager_load()


class EveryN(Processor):

    logger = logging.getLogger("EveryN")

    def __init__(
        self,
        processor: Processor,
        n: int,
        offset: int = 0,
        return_last: bool = True,
        override_condition: Optional[Callable[[Frame], bool]] = None,
        log: bool = False,
    ):
        self.processor = processor
        self.return_last = return_last
        self.override_condition = override_condition
        self.log = log
        self.n = n
        self.i = offset - 1
        self.last = False

    def run(self, frame: Frame) -> ProcessorTracing:
        t0 = time.perf_counter()
        processor_metadata = {}
        subprocessors = []

        self.i += 1
        run = False
        force = False
        if self.i % self.n == 0:
            run = True
        elif self.override_condition and self.override_condition(frame):
            run = True
            force = True
            processor_metadata["force_run"] = True

        if run:
            if self.log:
                self.logger.info(f"EverrN({self.processor}) {self.i} / {self.n} or {force} => True")
            subprocessors.append(self.processor.run(frame))
            r = subprocessors[-1].result
        elif self.return_last:
            if self.log:
                self.logger.info(f"EverrN({self.processor}) {self.i} / {self.n} => False")
            r = self.last
            processor_metadata["return_last"] = True
        else:
            if self.log:
                self.logger.info(f"EverrN({self.processor}) {self.i} / {self.n} => False")
            r = False
        return ProcessorTracing(
            name=self.__class__.__name__,
            qualname=self.__class__.__qualname__,
            duration=round(time.perf_counter() - t0, 4),
            result=r,
            subprocessors=subprocessors,
            processor_metadata=processor_metadata,
        )

    def eager_load(self):
        self.processor.eager_load()

    def update(self):
        self.processor.update()


# class RepeatProcessor(Processor):
#
#     def process(self, frame: Frame) -> bool:
#         if do process:
#             f = Frame(frame)
#             result = self.processor.process(frame)
#             if self.only_cache_on_positive:
#                 if result:
#                     self.cached = frame - f
#                 else:
#                     self.cached = None
#             else:
#
#         else:
#             frame.update(self.cached)

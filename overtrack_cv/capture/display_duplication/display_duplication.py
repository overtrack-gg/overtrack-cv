import logging
import os
import time
from ctypes import wintypes
from typing import List, NamedTuple

import cv2
import numpy as np
import psutil
import win32api
import win32gui
import win32process

from overtrack_cv.capture.display_duplication.native import *
from overtrack_cv.capture.display_duplication.source import *
from overtrack_cv.frame import Frame
from overtrack_cv.util.logging_config import config_logger, intermittent_log

logger = logging.getLogger("DisplayDuplicationCapture")

log_callback = ctypes.CFUNCTYPE(None, ctypes.c_char_p)
dll = ctypes.WinDLL(os.path.join(os.path.dirname(__file__), "ScreenCapture.dll"))
dll.init.argtypes = (ctypes.c_int, ctypes.c_int)
dll.init.restype = ctypes.c_int
dll.get_width.argtypes = ()
dll.get_width.restype = ctypes.c_long
dll.get_height.argtypes = ()
dll.get_height.restype = ctypes.c_long
dll.capture.argtypes = ()
dll.capture.restype = ctypes.c_void_p
dll.deinit.argtypes = ()
dll.deinit.restype = ctypes.c_int
dll.add_log_callback.argtypes = (log_callback,)
dll.add_log_callback.restype = None

dll_lasterror = None

# Use a pure python function with __module__ == 'screen_capture' to help logging when cythonized and logwrapped
dlllogger = logging.getLogger("ScreenCapture.dll")


def dll_log(s):
    line = s.decode("ascii")[:-1]
    if line.startswith("Error:"):
        global dll_lasterror
        dll_lasterror = line
    dlllogger.info(line)


@log_callback
def _dll_log(s):
    dll_log(s)


dll.add_log_callback(_dll_log)


def get_render_area(height, width, left_border_width=8, log=False):
    width_from_height = height * (1920 / 1080)
    height_from_width = width / (1920 / 1080)

    def info(*args):
        if log:
            logger.info(*args)

    if width_from_height == width:
        info("Game has no black bars")
        frame_y, render_height, frame_x, render_width = 0, height, left_border_width, width
    elif width_from_height > width:
        info("Game has black bars at the top and bottom - using client width as render width")
        render_width = width
        render_height = int(height_from_width)
        info("Render width = %d -> render height = %d" % (render_width, render_height))
        top_black_bar = int((height - render_height) / 2)
        frame_y = top_black_bar
        frame_x = left_border_width
        info(
            "Client height is %d and render height is %d -> top black bar height = %d",
            height,
            render_height,
            top_black_bar,
        )
    else:
        info("Game has black bars on the sides - using client height as render height")
        render_height = height
        render_width = int(width_from_height)
        info("Render height = %d -> render width = %d" % (render_height, render_width))
        frame_y = 0
        frame_x = int((width - render_width) / 2) + left_border_width
        info(
            "Client width is %d and render width is %d -> left black bar width = %d"
            % (width, render_width, frame_x)
        )

    info("Crop area: %d %d %d %d" % (frame_y, render_height, frame_x, render_width))
    return frame_y, render_height, frame_x, render_width


def resize_to_1080(screen, crop_bars: bool, interpolation=cv2.INTER_NEAREST, respect_width=True):
    if crop_bars:
        gimg = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        hline = np.ravel(cv2.reduce(gimg, 0, cv2.REDUCE_SUM, dtype=cv2.CV_32S)) > 0
        vline = np.ravel(cv2.reduce(gimg[:, 500:], 1, cv2.REDUCE_SUM, dtype=cv2.CV_32S)) > 0
        topbar = np.argmax(vline)
        if topbar > 5:
            bottombar = np.argmax(vline[-5::-1]) + 4
        else:
            bottombar = np.argmax(vline[::-1])
        leftbar, rightbar = np.argmax(hline), np.argmax(hline[::-1])

        if screen.shape[0] - (topbar + bottombar) > 250 and screen.shape[1] - (leftbar + rightbar) > 250:
            screen = screen[
                topbar : screen.shape[0] - bottombar,
                leftbar : screen.shape[1] - rightbar,
            ]

    if respect_width:
        if screen.shape[0] == 1080:
            return screen

        scale = 1080 / screen.shape[0]
        if screen.shape[1] * scale > 2560:
            scale = 2560 / screen.shape[1]

        cscale = min(scale, 2)
        scaled = cv2.resize(screen, (0, 0), fx=cscale, fy=cscale, interpolation=interpolation)
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
    else:
        return cv2.resize(screen, (1920, 1080), interpolation=interpolation)


zimg = np.zeros((1080, 1920, 4), dtype=np.uint8)


class Rect(NamedTuple):
    left: int
    top: int
    right: int
    bottom: int

    @classmethod
    def from_RECT(cls, rect: ctypes.wintypes.RECT):
        return Rect(rect.left, rect.top, rect.right, rect.bottom)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.right) // 2, (self.top + self.bottom) // 2

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def __str__(self) -> str:
        return (
            f"Rect(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom} | "
            f"width={self.width}, height={self.height})"
        )


def _get_monitors() -> List[Rect]:
    monitors = []

    def _enum_monitor(hMonitor, hdcMonitor, rect, dwData):
        monitors.append(Rect.from_RECT(rect[0]))
        return True

    ctypes.windll.user32.EnumDisplayMonitors(None, None, MonitorEnumProc(_enum_monitor), 0)
    intermittent_log(
        logger,
        f"Got monitors: {monitors}",
        frequency=1500,
        level=logging.INFO,
        negative_level=logging.DEBUG,
        caller_extra_id=(tuple([str(m) for m in monitors]),),
    )
    return monitors


def _get_monitor_containing(point: Tuple[int, int], monitors: List[Rect]) -> Optional[Rect]:
    monitor = None
    for m in monitors:
        if m.left < point[0] < m.right and m.top < point[1] < m.bottom:
            monitor = m
            break
    return monitor


def _fix_size(screen):
    if screen.shape == (768, 1366, 4) or screen.shape == (768, 1360, 4):
        screen = screen.flatten()
        screen.resize((765 * 1376 * 4,))
        screen = screen.reshape((765, 1376, 4))[1:, :-16]
    elif screen.shape == (1050, 1680, 4):
        screen = screen.flatten()
        screen.resize((1050 * 1696 * 4,))
        screen = screen.reshape((1050, 1696, 4))[:, :-16]
    elif screen.shape == (1440, 3440, 4):
        screen = screen.flatten()
        screen.resize((1440 * 3456 * 4,))
        screen = screen.reshape((1440, 3456, 4))[:, :-16]
    return screen


class ProcessNotFoundError(ValueError):
    pass


def get_hwnd(window: Optional[str], executable: Optional[str]) -> Tuple[wintypes.HWND, psutil.Process]:
    hwnd = None
    for attempt in range(100):
        # Look for all windows after `window` that match text
        # Because we update window each time, this iterates all the windows
        hwnd = win32gui.FindWindowEx(None, hwnd, None, window)
        if not hwnd:
            break

        try:
            _, process_id = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(process_id)

            if executable:
                logger.debug(f"Checking {hwnd} > {process} > {process.name().lower()} / {executable}")
                if process.name().lower() == executable.lower():
                    # try:
                    #     exe = process.exe()
                    # except psutil.AccessDenied as e:
                    #     logger.exception('Failed to get process exe')
                    #     exe = 'ACCESS DENIED'
                    # logger.info(f'Found HWND {hwnd} with name {window} - {process} exe={exe}')
                    logger.info(f"Found HWND {hwnd} with name {window} - {process}")
                    return hwnd, process
            else:
                logger.info(f"Found HWND {hwnd} with name {window} - {process}")
                return hwnd, process
        except Exception as e:
            logger.warning(f"Unable to find process for hwnd={hwnd}: {e}")

    raise ProcessNotFoundError(f"Could not window containing {window}")


class Crop(NamedTuple):
    top: int
    bottom: int
    left: int
    right: int

    pad_top: int = 0
    pad_bottom: int = 0
    pad_left: int = 0
    pad_right: int = 0

    def apply(self, img: np.ndarray) -> np.ndarray:
        img = img[self.top : self.bottom, self.left : self.right]
        if any((self.pad_top, self.pad_bottom, self.pad_left, self.pad_right)):
            img = cv2.copyMakeBorder(
                img,
                self.pad_top,
                self.pad_bottom,
                self.pad_left,
                self.pad_right,
                cv2.BORDER_CONSTANT,
                value=(30, 0, 30),
            )
        return img


class DisplayDuplicationCapture:

    CLIP_WHITELIST = ["overlay", "launcher", "engine", "d3dproxy"]

    _active_instance = None

    def __init__(
        self,
        hwnd: Optional[wintypes.HWND] = None,
        window_title: Optional[str] = None,
        executable: Optional[str] = None,
        foreground_window_pid: Optional[int] = None,
        debug_frames: bool = False,
    ):
        if self.__class__._active_instance:
            raise ValueError(f"Only one instance of {self.__class__.__name__} can be active at any time")

        self.destroyed = False
        self.debug_frames = debug_frames

        self.hwnd: Optional[wintypes.HWND]
        if hwnd:
            if window_title or executable:
                raise ValueError(f"Cannot provide a HWND along with a window/executable filter")
            if foreground_window_pid:
                raise ValueError(f"Cannot provide a HWND along with a foreground window PID filter")
            logger.info(f"Creating {self.__class__.__name__} with defined HWND {hwnd}")
            self.hwnd = hwnd
            self.foreground_window_pid = None
            self.window_title = None
            self.executable = None
        elif foreground_window_pid:
            if window_title or executable:
                raise ValueError(f"Cannot provide a HWND along with a window/executable filter")
            logger.info(f"Creating {self.__class__.__name__} with foreground PID filter {foreground_window_pid}")
            self.hwnd = None
            self.foreground_window_pid = foreground_window_pid
            self.window_title = None
            self.executable = None
        else:
            logger.info(
                f"Creating {self.__class__.__name__} with window/exe filter {window_title!r} / {executable!r}"
            )
            self.hwnd = None
            self.foreground_window_pid = None
            self.window_title = window_title
            self.executable = executable

        self.process: Optional[psutil.Process] = None

        self.monitor: Optional[Rect] = None

        self.style: Optional[int] = None
        self.ex_style: Optional[int] = None
        self.win_rect: Optional[Rect] = None
        self.client_rect: Optional[Rect] = None
        self.true_rect: Optional[Rect] = None
        self.window_info: Optional[WINDOWINFO] = None

        self.foreground = False
        self.fullscreen = False
        self.borderless = False
        self.maximised = False

        self.crop = None

        self.source = None

        self.ready = False
        self.backoff = 0
        self.last_reset: Optional[float] = None
        self.created = time.time()
        self._reset()

    def _make_crop(self) -> Optional[Crop]:
        if self.fullscreen or self.borderless:
            return None
        else:
            if dpi_detection:
                mon = ctypes.windll.user32.MonitorFromWindow(self.hwnd, 1)
                dpiX, dpiY = ctypes.wintypes.UINT(), ctypes.wintypes.UINT()
                ctypes.windll.shcore.GetDpiForMonitor(mon, 0, ctypes.byref(dpiX), ctypes.byref(dpiY))
                scale = dpiX.value / 96
            else:
                scale = 11

            if self.maximised:
                title_height = int(ctypes.windll.user32.GetSystemMetrics(SM_CYCAPTION) * scale + 0.5)
                x_offset = 0
            else:
                title_height = int(
                    (
                        win32api.GetSystemMetrics(SM_CYFRAME)
                        + win32api.GetSystemMetrics(SM_CYCAPTION)
                        + win32api.GetSystemMetrics(SM_CXPADDEDBORDER)
                    )
                    * scale
                    + 0.5
                )
                x_offset = 1

            # Find window location in monitor-relative space
            screenspace_left = self.true_rect.left
            screenspace_top = self.true_rect.top + title_height
            screenspace_right = screenspace_left + self.true_rect.width - 1
            sceenspace_bottom = screenspace_top + self.true_rect.height - title_height - 1

            # If the window goes off this monitor, we need to move the crop back inside the bounds and pad
            if screenspace_left < self.monitor.left:
                pad_left = self.monitor.left - screenspace_left
                screenspace_left = self.monitor.left
            else:
                pad_left = 0
            left = (screenspace_left - self.monitor.left) + x_offset

            if screenspace_top < self.monitor.top:
                pad_top = self.monitor.top - screenspace_top
                screenspace_top = self.monitor.top
            else:
                pad_top = 0
            top = screenspace_top - self.monitor.top

            if screenspace_right > self.monitor.right:
                pad_right = screenspace_right - self.monitor.right
                screenspace_right = self.monitor.right
            else:
                pad_right = 0
            right = screenspace_right - self.monitor.left

            if sceenspace_bottom > self.monitor.bottom:
                pad_bottom = sceenspace_bottom - self.monitor.bottom
                sceenspace_bottom = self.monitor.bottom
            else:
                pad_bottom = 0
            bottom = sceenspace_bottom - self.monitor.top

            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0:
                logger.warning(f"Got invalid crop: {top}:+{height}, {left}:+{width}", exc_info=True)
                return None
            else:
                logger.info(f"Got crop: {top}:+{height}, {left}:+{width}")

            return Crop(
                top=top,
                bottom=bottom,
                left=left,
                right=right,
                pad_top=pad_top,
                pad_bottom=pad_bottom,
                pad_left=pad_left,
                pad_right=pad_right,
            )

    def _reset(self) -> bool:
        if dpi_detection:
            ctypes.windll.user32.SetThreadDpiAwarenessContext(-3)

        intermittent_log(
            logger,
            f"Resetting {self}",
            frequency=500,
            level=logging.INFO,
            negative_level=logging.DEBUG,
            caller_extra_id=(
                self.hwnd,
                self.window_title,
                self.executable,
                self.foreground_window_pid,
                self.process.pid if self.process else None,
            ),
        )
        self.ready = False
        self.last_reset = time.time()

        self.monitors = _get_monitors()

        if not self.hwnd or not win32gui.IsWindow(self.hwnd):
            if self.window_title or self.executable:
                intermittent_log(
                    logger,
                    f"HWND invalid - looking for window matching window_title={self.window_title}, executable={self.executable}",
                    frequency=1500,
                    level=logging.INFO,
                    negative_level=logging.DEBUG,
                    caller_extra_id=(self.hwnd, self.window_title, self.executable),
                )
                try:
                    self.hwnd, self.process = get_hwnd(self.window_title, self.executable)
                except ProcessNotFoundError as e:
                    logger.warning(f"Could not find window: {e}")
                    self.hwnd, self.process = None, None
                    return False
            elif self.foreground_window_pid:
                intermittent_log(
                    logger,
                    f"HWND invalid - looking for matching foreground window with PID={self.foreground_window_pid}",
                    frequency=1500,
                    level=logging.INFO,
                    negative_level=logging.DEBUG,
                    caller_extra_id=(self.foreground_window_pid,),
                )
                foreground_hwnd = win32gui.GetForegroundWindow()
                _, foreground_process_id = win32process.GetWindowThreadProcessId(foreground_hwnd)
                if foreground_process_id == self.foreground_window_pid:
                    logger.info(f"Found foreground HWND matching PID: {foreground_hwnd}")
                    self.hwnd = foreground_hwnd
                else:
                    intermittent_log(
                        logger,
                        f"Could not find foreground window matching PID",
                        frequency=1500,
                        level=logging.INFO,
                        negative_level=logging.DEBUG,
                        caller_extra_id=(self.foreground_window_pid,),
                    )
                    self.hwnd = False
                    return False

        if self.hwnd:
            self.style = win32api.GetWindowLong(self.hwnd, GWL_STYLE)
            logger.info(f"Got GWL_STYLE={self.style:08x}")

            self.ex_style = win32api.GetWindowLong(self.hwnd, GWL_EXSTYLE)
            logger.info(f"Got GWL_EXSTYLE={self.ex_style:08x}")

            self.win_rect = Rect(*win32gui.GetWindowRect(self.hwnd))
            logger.info(f"Got WindowRect={self.win_rect}")

            self.client_rect = Rect(*win32gui.GetClientRect(self.hwnd))
            logger.info(f"Got ClientRect={self.client_rect}")

            try:
                self.true_rect = Rect.from_RECT(DwmGetWindowAttribute(self.hwnd))
                logger.info(f"Got true_rect={self.true_rect}")
            except Exception as e:
                logger.warning(f"Failed to get true_rect (DwmGetWindowAttribute): {e} - using GetWindowRect")
                self.true_rect = self.win_rect

            self.window_info = GetWindowInfo(self.hwnd)
            logger.info(f"Got WindowInfo={self.window_info}")

            self.foreground = False
            if not self.style & WS_MINIMIZE:
                center = self.win_rect.center
                if center:
                    logger.info(f"Looking for monitor containing {center}")
                    self.monitor = _get_monitor_containing(center, self.monitors)
                    logger.info(f"Got monitor={self.monitor}")
                    if self.monitor:
                        self.foreground = True

            self.fullscreen = bool(self.ex_style & WS_EX_TOPMOST)
            self.borderless = bool(not self.style & WS_BORDER)
            if self.fullscreen or self.borderless:
                self.maximised = True
            else:
                self.maximised = bool(self.style & WS_MAXIMIZE)

            if self.foreground:
                self.crop = self._make_crop()
                logger.info(f"Got crop: {self.crop}")

        else:
            # no window to track - just use first monitor
            self.monitor = self.monitors[0]

            self.foreground = True
            self.fullscreen = True
            self.borderless = True
            self.maximised = True

            self.crop = None

        self.source = self._make_source()

        if self.foreground:
            monitor_center = self.monitor.center

            dll.deinit()
            logger.info(f"Trying to get monitor capture for center={monitor_center}")
            if dll.init(int(monitor_center[0]), int(monitor_center[1])):
                logger.warning("ScreenCapture.dll failed to init")
                return False

            bufaddr = dll.capture()
            if not bufaddr:
                logger.warning("ScreenCapture.dll capture failed to return image data")
                return False

            buffsize = dll.get_height() * dll.get_width() * 4
            d11buf = PyMemoryView_FromMemory(bufaddr, buffsize, PyBUF_READ)
            self.img = np.ndarray((dll.get_height(), dll.get_width(), 4), np.uint8, d11buf, order="C")
            logger.info("Done resetting DirectXCapture")

            self.ready = True
            return True
        else:
            self.ready = True
            return True

    def _clip_windows(self, screen):
        if not ctypes.windll.user32.IsWindowVisible(self.hwnd):
            # overwatch window not visible
            screen[:] = 50
            return screen

        clip_hwnd = win32gui.GetWindow(self.hwnd, GW_HWNDPREV)
        while clip_hwnd:

            try:
                other_rect = Rect(*win32gui.GetWindowRect(clip_hwnd))

                name = win32gui.GetWindowText(clip_hwnd)

                clip_window = True
                for whitelist_app in self.CLIP_WHITELIST:
                    if whitelist_app.lower() in name.lower():
                        # logger.warning(f'Not clipping {name.value}')
                        clip_window = False

                if (
                    clip_window
                    and win32gui.IsWindowVisible(clip_hwnd)
                    and other_rect.width > 0
                    and other_rect.height > 0
                    and name
                ):
                    monitorspace_rect = Rect(
                        left=max(0, other_rect.left - self.monitor.left),
                        top=max(0, other_rect.top - self.monitor.top),
                        right=max(0, min(other_rect.right - self.monitor.left, self.monitor.width)),
                        bottom=max(0, min(other_rect.bottom - self.monitor.top, self.monitor.height)),
                    )
                    # logger.debug(f'Clipping {name.value} | {self.win_rect}, {other_rect} => {monitorspace_rect}')
                    if monitorspace_rect.width > 10 and monitorspace_rect.height > 10:
                        screen[
                            monitorspace_rect.top : monitorspace_rect.bottom,
                            monitorspace_rect.left : monitorspace_rect.right,
                        ] = 0
                        screen = cv2.putText(
                            screen,
                            name,
                            (monitorspace_rect.left + 100, monitorspace_rect.top + 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )
                    # else:
                    #     logger.debug(f'Clipping {name.value} | {self.win_rect}, {other_rect} => None')
            except:
                logger.exception(f"Failed to clip window")

            clip_hwnd = win32gui.GetWindow(clip_hwnd, GW_HWNDPREV)
        return screen

    def check_needs_reset(self):
        if self.hwnd:
            if not win32gui.IsWindow(self.hwnd):
                logger.warning(f"HWND became invalid - resetting")
                return True

            style = win32api.GetWindowLong(self.hwnd, GWL_STYLE)
            if style != self.style:
                logger.warning(f"GWL_STYLE changed - resetting")
                return True

            ex_style = win32api.GetWindowLong(self.hwnd, GWL_EXSTYLE)
            if ex_style != self.ex_style:
                logger.warning(f"GWL_EXSTYLE changed - resetting")
                return True

            win_rect = win32gui.GetWindowRect(self.hwnd)
            if win_rect != self.win_rect:
                logger.warning(f"WindowRect changed - resetting")
                return True

            client_rect = win32gui.GetClientRect(self.hwnd)
            if client_rect != self.client_rect:
                logger.warning(f"ClientRect changed - resetting")
                return True

            if self.foreground:
                monitor = _get_monitor_containing(self.win_rect.center, self.monitors)
                if not monitor or self.monitor != monitor:
                    logger.warning(f"Monitor containing center changed - resetting")
                    return True

        return False

    def capture(self, interpolation=cv2.INTER_LINEAR):
        if self.destroyed:
            raise ValueError(f"Cannot capture from destroyed {self.__class__.__name__}")

        if not self.ready:
            since_last = time.time() - self.last_reset
            if since_last > self.backoff:
                if self._reset():
                    self.backoff = 0
                else:
                    self.backoff = min(max(1, self.backoff) * 2, 10)
                    intermittent_log(
                        logger,
                        f"_reset failed - retrying in {self.backoff}s",
                        frequency=500,
                        level=logging.WARNING,
                        negative_level=logging.DEBUG,
                        caller_extra_id=(self.hwnd, self.backoff),
                    )
                    return zimg, False
            else:
                return zimg, False

        if dpi_detection:
            ctypes.windll.user32.SetThreadDpiAwarenessContext(-3)

        if self.check_needs_reset():
            self.ready = False
            return zimg, False
        elif self.foreground:
            if not dll.capture():
                logger.warning(f"Monitor capture failed - resetting")
                self.ready = False
                return zimg, False
            else:
                screen = self.img.copy()
                screen = _fix_size(screen)

                if not self.fullscreen:
                    screen = self._clip_windows(screen)

                client_area = screen
                if self.crop:
                    client_area = self.crop.apply(client_area)

                crop_bars = self.maximised and not self.borderless
                return resize_to_1080(client_area, crop_bars, interpolation=interpolation), True
        else:
            return zimg, False

    def get(self) -> Frame:
        start = time.time()
        img, valid = self.capture()
        end = time.time()

        if img.shape != (1080, 1920, 4):

            if img.shape[0] > 1080:
                img = img[:1080]
                bpad = 0
            else:
                bpad = 1080 - img.shape[0]

            if img.shape[1] > 1920:
                img = img[:, :1920]
                rpad = 0
            else:
                rpad = 1920 - img.shape[1]

            if rpad or bpad:
                img = cv2.copyMakeBorder(img, 0, bpad, 0, rpad, cv2.BORDER_CONSTANT)

        f = Frame.create(
            img[:, :, :3],
            timestamp=round(start, 2),
            relative_timestamp=round(start - self.created, 2),
            debug=self.debug_frames,
            valid=valid,
            foreground=self.foreground,
            image_alpha=img[:, :, 3],
            source=self.source,
            source_name="Display Duplication",
        )
        # f.timings[self.__class__.__name__] = (end - start) * 1000
        return f

    def close(self):
        self.ready = False
        dll.deinit()
        self.destroyed = True
        self.__class__._active_instance = None

    def __str__(self) -> str:
        if self.foreground_window_pid:
            filterstr = f"foreground_window_pid={self.foreground_window_pid}"
        elif self.window_title or self.executable:
            filterstr = f"(window_title={self.window_title!r}, executable={self.executable})"
        else:
            filterstr = None
        return (
            f"{self.__class__.__name__}("
            f"filter={filterstr}, "
            f"hwnd={self.hwnd}, "
            f"foreground={self.foreground}, "
            f"fullscreen={self.fullscreen}, "
            f"borderless={self.borderless}, "
            f"maximised={self.maximised}"
            f")"
        )

    def _make_source(self) -> DisplayDuplicationSource:
        # exe, cmdline = None, None
        # if self.process:
        #     try:
        #         exe = str(self.process.exe())
        #     except psutil.Error as e:
        #         exe = f'!{e}'
        #     try:
        #         cmdline = str(self.process.cmdline())
        #     except psutil.Error as e:
        #         cmdline = f'!{e}'

        if self.hwnd is None:
            name = "DESKTOP"
        elif self.process is None:
            name = f"hwnd={self.hwnd}"
        else:
            try:
                name = self.process.name()
            except psutil.Error as e:
                name = f"!{e}"

        return DisplayDuplicationSource(
            hwnd=int(self.hwnd) if self.hwnd else None,
            pid=self.process.pid if self.process else None,
            name=name,
            window_title=self.window_title,
            style=int(self.style) if self.style else None,
            ex_style=int(self.ex_style) if self.ex_style else None,
            rect=tuple(self.true_rect) if self.true_rect else None,
            monitor=tuple(self.monitor) if self.monitor else None,
        )


def get_error_hint(msg: str) -> str:
    errorcode = msg.split(" ")[-1].strip()
    if errorcode[-1] == ".":
        errorcode = errorcode[:-1].strip()

    if errorcode == "0x887A0022":
        return "Not Currently Available:\nYou may need to close other programs recording your screen"
    elif errorcode == "0x80070005":
        return "Access Denied:\nYou may need to run OverTrack as Admin"
    elif errorcode in ["0x887A0004", "0x80070057"]:
        return "Unsupported by Graphics Driver:\nYou may need to run OverTrack on the a different GPU, or your device may not be supported"

    return ""


def main() -> None:
    # hwnd, pid = get_hwnd('Overwatch', None)
    # hwnd, pid = get_hwnd(None, 'pycharm64.exe')
    # print(hwnd, pid)
    from overtrack_models.dataclasses.typedload import referenced_typedload

    # d = DisplayDuplicationCapture(window_title='Overwatch', executable='overwatch.exe')
    title = "Apex Legends"
    executable = "r5apex.exe"
    d = DisplayDuplicationCapture(window_title=title, executable=executable)
    while True:
        frame = d.get()
        cv2.imshow("frame", frame.image)
        frame.strip()
        print(frame.valid)
        cv2.waitKey(1000)

        referenced_typedload.dump(frame)


if __name__ == "__main__":

    config_logger("display_duplication", logging.DEBUG, False)
    main()

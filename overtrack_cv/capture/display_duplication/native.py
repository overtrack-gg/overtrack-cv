import ctypes
from ctypes import wintypes

dwmapi = ctypes.WinDLL("dwmapi")

LPRECT = ctypes.POINTER(wintypes.RECT)
MonitorEnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HMONITOR, wintypes.HDC, LPRECT, wintypes.LPARAM)
ctypes.windll.user32.EnumDisplayMonitors.restype = wintypes.BOOL
ctypes.windll.user32.EnumDisplayMonitors.argtypes = (wintypes.HDC, LPRECT, MonitorEnumProc, wintypes.LPARAM)

ctypes.windll.user32.GetWindow.argtypes = (wintypes.HWND, ctypes.c_int)
ctypes.windll.user32.GetWindow.restype = wintypes.HWND
GW_HWNDNEXT = 2
GW_HWNDPREV = 3

ctypes.windll.user32.GetSystemMetrics.argtypes = (ctypes.c_int,)
ctypes.windll.user32.GetSystemMetrics.restype = ctypes.c_int
SM_CYCAPTION = 4
SM_CXBORDER = 5
SM_CYSIZE = 31
SM_CXFRAME = 32
SM_CYFRAME = 33
SM_CXPADDEDBORDER = 92

GWL_EXSTYLE = -20
GWL_STYLE = -16

WS_MINIMIZE = 0x20000000
WS_BORDER = 0x00800000
WS_MAXIMIZE = 0x01000000
WS_EX_TOPMOST = 0x00000008

try:
    ctypes.windll.shcore.GetDpiForMonitor.argtypes = (
        wintypes.HMONITOR,
        ctypes.c_int,
        ctypes.POINTER(wintypes.UINT),
        ctypes.POINTER(wintypes.UINT),
    )
    ctypes.windll.shcore.GetDpiForMonitor.restype = ctypes.c_int

    ctypes.windll.user32.SetThreadDpiAwarenessContext.argtypes = (ctypes.c_int,)
    ctypes.windll.user32.SetThreadDpiAwarenessContext.restype = ctypes.c_int
except Exception as e:
    dpi_detection = False
else:
    dpi_detection = True

PyMemoryView_FromMemory = ctypes.pythonapi.PyMemoryView_FromMemory
PyMemoryView_FromMemory.restype = ctypes.py_object
PyMemoryView_FromMemory.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
PyBUF_READ = 0x100

dwmapi.DwmGetWindowAttribute.restype = ctypes.HRESULT
dwmapi.DwmGetWindowAttribute.argtypes = (wintypes.HWND, wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD)
DWMWA_EXTENDED_FRAME_BOUNDS = 9


def DwmGetWindowAttribute(hwnd: wintypes.HWND) -> wintypes.RECT:
    rect = wintypes.RECT()
    dwmapi.DwmGetWindowAttribute(hwnd, DWMWA_EXTENDED_FRAME_BOUNDS, ctypes.byref(rect), ctypes.sizeof(rect))
    return rect


class WINDOWINFO(ctypes.Structure):
    def __str__(self) -> str:
        return (
            "WINDOWINFO(" + ", ".join([key + ":" + str(getattr(self, key)) for key, value in self._fields_]) + ")"
        )


WINDOWINFO._fields_ = [
    ("cbSize", wintypes.DWORD),
    ("rcWindow", wintypes.RECT),
    ("rcClient", wintypes.RECT),
    ("dwStyle", wintypes.DWORD),
    ("dwExStyle", wintypes.DWORD),
    ("dwWindowStatus", wintypes.DWORD),
    ("cxWindowBorders", wintypes.UINT),
    ("cyWindowBorders", wintypes.UINT),
    ("atomWindowType", wintypes.ATOM),
    ("wCreatorVersion", wintypes.WORD),
]


def GetWindowInfo(hwnd: wintypes.HWND) -> WINDOWINFO:
    pwi = WINDOWINFO()
    ctypes.windll.user32.GetWindowInfo(hwnd, ctypes.byref(pwi))
    return pwi

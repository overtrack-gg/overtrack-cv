import inspect
import logging
import logging.config
import os
import sys
import time
from collections import defaultdict
from threading import Thread
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Optional,
    Sequence,
    Tuple,
)

if TYPE_CHECKING:
    from typing_extensions import TypedDict

    LogConfig = TypedDict(
        "LogConfig",
        {
            "level": str,
            "formatter": str,
            "class": str,
            "filename": str,
            "maxBytes": int,
            "backupCount": int,
            "delay": bool,
        },
        total=False,
    )
    UploadLogsSettingsType = TypedDict(
        "UploadLogsSettingsType",
        {
            "write_to_file": bool,
            "upload_func": Callable[[str, str], None],
            "args": Tuple[str, str],
        },
        total=False,
    )
else:
    LogConfig = Dict
    UploadLogsSettingsType = Dict


LOG_FORMAT = "[%(asctime)16s | %(levelname)8s | %(name)24s | %(filename)s:%(lineno)s %(funcName)s() ] %(message)s"


def intermittent_log(
    logger: logging.Logger,
    line: str,
    frequency: float = 60,
    level: int = logging.INFO,
    negative_level: Optional[int] = None,
    caller_extra_id: Any = None,
    fn_override: Optional[str] = None,
    line_override: Optional[int] = None,
    func_override: Optional[str] = None,
    _caller: Optional[str] = None,
    _last_logged: DefaultDict[Tuple[str, int], float] = defaultdict(float),
    _times_suppressed: DefaultDict[Tuple[str, int], float] = defaultdict(int),
) -> None:
    try:
        caller = None
        if _caller:
            frame_id = _caller, caller_extra_id
        else:
            try:
                caller = inspect.stack()[1]
                frame_id = caller.filename, caller.lineno, caller_extra_id
            except:
                frame_id = "???"

        output = negative_level
        if time.time() - _last_logged[frame_id] > frequency:
            _last_logged[frame_id] = time.time()
            if _times_suppressed[frame_id]:
                line += f" [log suppressed {_times_suppressed[frame_id]} times since last]"
            _times_suppressed[frame_id] = 0
            output = level
        else:
            _times_suppressed[frame_id] += 1
        if output and logger.isEnabledFor(output):
            if caller:
                co = caller.frame.f_code
                fn, lno, func, sinfo = (
                    co.co_filename,
                    caller.frame.f_lineno,
                    co.co_name,
                    None,
                )
                record = logger.makeRecord(
                    logger.name,
                    output,
                    fn_override or str(fn),
                    line_override or lno,
                    line,
                    {},
                    None,
                    func_override or func,
                    None,
                    sinfo,
                )
            else:
                record = logger.makeRecord(
                    logger.name,
                    output,
                    fn_override or "???",
                    line_override or 0,
                    line,
                    {},
                    None,
                    func_override,
                )
            logger.handle(record)
    except:
        # noinspection PyProtectedMember
        logger.log(level, line, ())


upload_logs_settings: UploadLogsSettingsType = {
    "write_to_file": False,
}


def logname(s: str) -> str:
    return str(os.path.basename(s)).rsplit(".", 1)[0]


def config_logger(
    name: str,
    level: int = logging.INFO,
    write_to_file: bool = True,
    use_stackdriver: bool = False,
    stackdriver_level: int = logging.INFO,
    stackdriver_name: Optional[str] = None,
    tracemalloc: bool = False,
    upload_func: Optional[Callable[[str, str], None]] = None,
    upload_frequency: Optional[float] = None,
    custom_loggers_config: Optional[Dict[str, Dict]] = None,
    format: str = LOG_FORMAT,
    logdir: str = "logs",
) -> None:

    logger = logging.getLogger()

    if name.endswith(".py"):
        name = name.rsplit(".")[0]

    handlers: Dict[str, LogConfig] = {
        "default": {
            "level": logging.getLevelName(level),
            "formatter": "standard",
            "class": "logging.StreamHandler",
        }
    }
    if write_to_file:
        os.makedirs(logdir, exist_ok=True)
        handlers.update(
            {
                "file": {
                    "level": "INFO",
                    "formatter": "standard",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": f"{logdir}/{name}.log",
                    "maxBytes": 1024 * 1024 * 100,
                    "backupCount": 3,
                    "delay": True,
                },
                "file_debug": {
                    "level": "DEBUG",
                    "formatter": "standard",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": f"{logdir}/{name}.debug.log",
                    "maxBytes": 1024 * 1024 * 100,
                    "backupCount": 3,
                    "delay": True,
                },
                "web_access": {
                    "level": "DEBUG",
                    "formatter": "",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": f"{logdir}/access.log",
                    "maxBytes": 1024,
                    "backupCount": 0,
                    "delay": True,
                },
            }
        )
    else:
        handlers.update(
            {
                "file": {
                    "class": "logging.NullHandler",
                },
                "file_debug": {
                    "class": "logging.NullHandler",
                },
                "web_access": {
                    "class": "logging.NullHandler",
                },
            }
        )

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {"format": format},
            },
            "handlers": handlers,
            "loggers": {
                "": {
                    "handlers": ["default", "file", "file_debug"],
                    "level": "DEBUG",
                    "propagate": True,
                },
                "cherrypy.access": {
                    "handlers": ["web_access"],
                    "level": "WARN",
                    "propagate": False,
                },
                "sanic.access": {
                    "handlers": ["web_access"],
                    "level": "WARN",
                    "propagate": False,
                },
                "libav.AVBSFContext": {
                    "handlers": ["default", "file", "file_debug"],
                    "level": "CRITICAL",
                    "propagate": False,
                },
                "libav.swscaler": {
                    "handlers": ["default", "file", "file_debug"],
                    "level": "CRITICAL",
                    "propagate": False,
                },
                "datadog.api": {"handlers": [], "level": "ERROR", "propagate": False},
                **(custom_loggers_config or {}),
            },
        }
    )

    if use_stackdriver:
        import google.cloud.logging
        from google.cloud.logging.handlers import CloudLoggingHandler
        from google.cloud.logging.handlers.handlers import EXCLUDED_LOGGER_DEFAULTS

        # noinspection PyUnresolvedReferences
        client = google.cloud.logging.Client()
        # client.setup_logging()

        handler = CloudLoggingHandler(client, name=stackdriver_name or name)
        handler.setLevel(stackdriver_level)
        logger.addHandler(handler)
        for logger_name in EXCLUDED_LOGGER_DEFAULTS + ("urllib3.connectionpool",):
            exclude = logging.getLogger(logger_name)
            exclude.propagate = False
            # exclude.addHandler(logging.StreamHandler())

    if tracemalloc:
        import tracemalloc

        tracemalloc.start()

        tracemalloc_logger = logging.getLogger("tracemalloc")

        def tracemalloc_loop():
            while True:
                time.sleep(5 * 60)
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics("lineno")
                tracemalloc_logger.info(f"tracemalloc:")
                for stat in top_stats[:10]:
                    tracemalloc_logger.info(f"  {stat}")

        Thread(target=tracemalloc_loop, name="tracemalloc", daemon=True).start()

    # if use_stackdriver_error:
    #     from google.cloud import error_reporting
    #     client = error_reporting.Client()

    # if use_datadog:
    #     import datadog
    #     from datadog_logger import DatadogLogHandler
    #     datadog.initialize(api_key=os.environ['DATADOG_API_KEY'], app_key=os.environ['DATADOG_APP_KEY'])
    #     datadog_handler = DatadogLogHandler(
    #         tags=[
    #             f'host:{socket.gethostname()}',
    #             f'pid:{os.getpid()}',
    #             f'stack:{name}',
    #             'type:log'],
    #         mentions=[],
    #         level=logging.INFO
    #     )
    #     logger.addHandler(datadog_handler)

    for _ in range(3):
        logger.info("")
    logger.info(f'Command: "{" ".join(sys.argv)}", pid={os.getpid()}, name={name}')
    if use_stackdriver:
        logger.info(
            f"Connected to google cloud logging. Using name={name!r}. Logging class: {logging.getLoggerClass()}"
        )

    upload_logs_settings["write_to_file"] = write_to_file
    if write_to_file and upload_func and upload_frequency:
        upload_logs_settings["upload_func"] = upload_func
        file: str = handlers["file"]["filename"]
        file_debug: str = handlers["file_debug"]["filename"]
        # noinspection PyTypeChecker
        upload_logs_settings["args"] = file, file_debug

        def upload_loop() -> None:
            while True:
                assert upload_frequency
                assert upload_func
                time.sleep(upload_frequency)
                upload_func(handlers["file"]["filename"], handlers["file_debug"]["filename"])

        logger.info(f"Uploading log files every {upload_frequency}s")
        Thread(target=upload_loop, daemon=True).start()

    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    # hsh = hashlib.md5()
    # modules = [
    #     m.__file__ for m in globals().values() if
    #     isinstance(m, types.ModuleType) and
    #     hasattr(m, '__file__')
    # ]
    # modules.append(__file__)
    # for mod in sorted(modules):
    #     with open(mod, 'rb') as f:
    #         hsh.update(f.read())
    # logger.info(f'Modules hash: {hsh.hexdigest()}')


def finish_logging() -> None:
    if (
        upload_logs_settings.get("write_to_file")
        and upload_logs_settings.get("upload_func")
        and upload_logs_settings.get("args")
    ):
        upload_logs_settings["upload_func"](*upload_logs_settings["args"])


def patch_sentry_locals_capture() -> None:
    from sentry_sdk.serializer import Serializer, add_global_repr_processor
    from sentry_sdk.utils import safe_repr

    from overtrack_cv.frame import Frame

    ser = Serializer()

    def frame_processor(value, hint):
        if isinstance(value, Frame):
            from overtrack_models.dataclasses import typedload

            f = value.copy()
            f.strip()
            return typedload.dump(f)
        return NotImplemented

    def frame_list_processor(value, hint):
        if (
            isinstance(value, list)
            and len(value) > 2
            and isinstance(value[0], Frame)
            and isinstance(value[-1], Frame)
        ):
            return [
                frame_processor(value[0], None),
                f"...<{len(value) - 2}>...",
                frame_processor(value[-1], None),
            ]
        return NotImplemented

    def large_list_processor(value, hint):
        if isinstance(value, Sequence) and not isinstance(value, (bytes, str)) and len(value) > 64:
            return [
                ser._serialize_node_impl(value[0], None, None),
                f"...<{len(value) - 2}>...",
                ser._serialize_node_impl(value[-1], None, None),
            ]
        return NotImplemented

    def large_str_processor(value, hint):
        if isinstance(value, (bytes, str)) and len(value) > 1024:
            return safe_repr(value[:1024]) + f"..., len={len(value)}"
        return NotImplemented

    def set_processor(value, hint):
        if isinstance(value, set):
            return list(value)
        return NotImplemented

    add_global_repr_processor(frame_processor)
    add_global_repr_processor(frame_list_processor)
    add_global_repr_processor(large_list_processor)
    add_global_repr_processor(large_str_processor)
    add_global_repr_processor(set_processor)


CLOUD_INIT_OUTPUT = "/var/log/cloud-init-output.log"

# def log_cloud_init():
#     logger = logging.getLogger('cloud-init')
#     if os.path.exists(CLOUD_INIT_OUTPUT):
#         logger.info(f'Found cloud init log: {CLOUD_INIT_OUTPUT}')
#         logger.info('-' * 15 + ' Begin Cloud Init Log ' + '-' * 15)
#         with open(CLOUD_INIT_OUTPUT) as f:
#             started = False
#             for line in f.readlines():


def main() -> None:
    config_logger("adasd", level=logging.INFO)
    logger = logging.getLogger()
    logger.info("foo")


if __name__ == "__main__":
    main()

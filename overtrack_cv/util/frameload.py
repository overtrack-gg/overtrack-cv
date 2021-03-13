import base64
import bz2
from dataclasses import dataclass
from typing import Any, Dict, NamedTuple, Optional, Type, TypeVar, Union

from overtrack_cv_private.games.overwatch.overwatch_frame_data import OverwatchFrameData
from overtrack_cv_private.games.valorant.valorant_frame_data import ValorantFrameData
from overtrack_models.dataclasses.typedload.referenced_typedload import (
    ReferencedDumper,
    ReferencedLoader,
)

from overtrack_cv.capture.display_duplication import DisplayDuplicationSource
from overtrack_cv.capture.shmem import SharedMemorySource
from overtrack_cv.core.uploadable_image import UploadableImage, UploadedImage
from overtrack_cv.frame import CurrentGame, Frame, SerializableArray
from overtrack_cv.games.apex.apex_frame_data import ApexFrameData

T = TypeVar("T")


def frames_dump(value: object, ignore_default: bool = True, numpy_support: bool = True) -> Dict[str, Any]:
    return FrameDumper(ignore_default=ignore_default, numpy_support=numpy_support).dump(value)


def frames_load(value: Any, type_: Type[T]) -> T:
    return FrameLoader().load(value, type_)


class FrameDumper(ReferencedDumper):
    _dispatch = ReferencedDumper._dispatch.copy()

    def _is_uploadable_image(self, value) -> bool:
        return isinstance(value, UploadableImage)

    def _dump_uploadable_image(self, value: UploadableImage) -> Dict[Any, Any]:
        return value._typeddump()

    _dispatch.append((_is_uploadable_image, _dump_uploadable_image))

    def _is_serializable_array(self, value) -> bool:
        return isinstance(value, SerializableArray)

    def _dump_serializable_array(self, value: SerializableArray) -> Dict[Any, Any]:
        import numpy as np

        array = value.finalize()
        assert isinstance(array, np.ndarray)
        data = base64.b85encode(bz2.compress(array.tobytes())).decode()
        return {
            "dtype": str(array.dtype),
            "shape": array.shape,
            "compression": "bz2",
            "data": [data[i : i + 128] for i in range(0, len(data), 128)],
        }

    _dispatch.append((_is_serializable_array, _dump_serializable_array))


_Source = Union[SharedMemorySource, DisplayDuplicationSource]

try:
    from overtrack_cv.capture import UploadedTSSource

    _Source = Union[_Source, UploadedTSSource]
except ImportError:
    pass
try:
    from overtrack_cv.capture import HTTPSource

    _Source = Union[_Source, HTTPSource]
except ImportError:
    pass
try:
    from overtrack_cv.capture import TSSource

    _Source = Union[_Source, TSSource]
except ImportError:
    pass
try:
    from overtrack_cv.capture import TwitchSource

    _Source = Union[_Source, TwitchSource]
except ImportError:
    pass


class Segment(NamedTuple):
    uri: Optional[str]
    duration: float
    title: str
    key: Optional[Any]
    discontinuity: bool
    ad: Optional[bool]
    byterange: Optional[Any]
    date: Optional[str]
    map: Optional[Any]


class Sequence(NamedTuple):
    num: int
    segment: Segment


@dataclass
class FrameInfo:
    type: str
    type: str
    key_frame: int
    format: str
    dts: Optional[float]
    pts: Optional[float]


_TYPES = {
    "overwatch": OverwatchFrameData,
    "apex": ApexFrameData,
    "valorant": ValorantFrameData,
    "source": _Source,
    "current_game": CurrentGame,
    "sequence": Sequence,
    "frame_info": FrameInfo,
}
_REMAPS = {"objective2": "objective", "killfeed_2": "killfeed"}


class FrameLoader(ReferencedLoader):
    _dispatch = ReferencedLoader._dispatch.copy()

    def __init__(self):
        frefs = {}
        # for f in fields(ValorantFrameData):
        # 	typestr = f.type.__args__[0].__forward_arg__
        # 	modulestr, classname = typestr.rsplit('.', 1)
        # 	type_ = __import__(modulestr)
        # 	for p in typestr.split('.')[1:]:
        # 		type_ = getattr(type_, p)
        # 	frefs[typestr] = type_
        super().__init__(frefs)

    def _is_frame(self, type_: Type) -> bool:
        return type_ == Frame

    def _load_frame(self, value: Dict[str, Any], type_: Type[Frame]) -> Frame:
        if not isinstance(value, dict):
            raise TypeError(f"Could not load {type_} from value {value!r}")
        result = Frame.__new__(Frame)
        result.image = result.debug_image = result._image_yuv = None

        # properties to into the new GameProperties object
        game_data_fields = {"overwatch": {}, "apex": {}}

        for k, v in value.items():

            # support remapping keys
            k = _REMAPS.get(k, k)

            if k in ["image", "debug_image", "_image_yuv"]:
                continue

            # if the property name exists in the fields of OverwatchGameData of ApexGameData it needs to me moved onto that object
            for g in game_data_fields:
                if hasattr(_TYPES[g], k):
                    game_data_fields[g][k] = v
                    break
            else:
                if v is None or type(v) in {int, bool, float, str}:
                    result[k] = v
                elif k in _TYPES:
                    result[k] = self.load(v, _TYPES[k])
                elif k == "timings":
                    result[k] = v
                else:
                    raise TypeError(f"Don't know how to load Frame field {k!r}")

        # move properties
        for g in game_data_fields:
            if game_data_fields[g]:
                result[g] = self._load_dataclass(game_data_fields[g], _TYPES[g])
        for g in "overwatch", "apex", "valorant":
            if g not in result:
                setattr(result, g, _TYPES[g]())

        return result

    _dispatch.append((_is_frame, _load_frame))

    def _is_uploadable_image(self, type_: Type) -> bool:
        return type_ == UploadableImage

    def _load_uploadable_image(self, image: UploadableImage, type_: Type[UploadableImage]) -> UploadedImage:
        # Convert UploadableImage -> UploadedImage
        return self.load(image, UploadedImage)

    _dispatch.insert(0, (_is_uploadable_image, _load_uploadable_image))

    def _is_serializable_array(self, type_: Type) -> bool:
        return type_ == SerializableArray

    def _load_serializable_array(self, value: Dict[str, Any], type_: Type[SerializableArray]) -> SerializableArray:
        import numpy as np

        assert value["compression"] == "bz2"
        string = "".join(value["data"])
        data = bz2.decompress(base64.b85decode(string))
        array = np.fromstring(data, dtype=np.dtype(value["dtype"])).reshape(value["shape"])
        return SerializableArray(array)

    _dispatch.append((_is_serializable_array, _load_serializable_array))

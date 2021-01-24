import inspect
import logging
import operator
import os
import string
from threading import Lock
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    no_type_check,
    overload,
)

import cv2
import numpy as np
import tesserocr

if TYPE_CHECKING:
    from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
    from overtrack_cv.frame import Frame
else:
    ExtractionRegionsCollection = Frame = object


logger = logging.getLogger(__name__)


class ConnectedComponent(NamedTuple):

    label: int
    x: int
    y: int
    w: int
    h: int

    area: int
    centroid: Tuple[float, float]


def connected_components(image: np.ndarray, connectivity: int = 4) -> Tuple[np.ndarray, List[ConnectedComponent]]:
    r, labels, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=connectivity)
    components = []
    for i, (stat, centroid) in enumerate(zip(stats, centroids)):
        # thanks mypy :/
        components.append(
            ConnectedComponent(
                i,
                int(stat[0]),
                int(stat[1]),
                int(stat[2]),
                int(stat[3]),
                int(stat[4]),
                centroid=(float(centroid[0]), float(centroid[1])),
            )
        )
    return labels, components


def otsu_thresh(vals: np.ndarray, mn: float, mx: float) -> float:
    # adapted from https://github.com/scikit-image/scikit-image/blob/v0.14.0/skimage/filters/thresholding.py#L230: threshold_otsu

    mnv = np.clip(mn, 0, 253)
    mxv = np.clip(mx, mn + 2, 255)
    hist, bin_edges = np.histogram(vals, int(mxv - mnv), (mnv, mxv + 1))
    histv = hist.astype(float)
    bin_edges = bin_edges[1:]

    # class probabilities for all possible thresholds
    weight1 = np.cumsum(histv)
    weight2 = np.cumsum(histv[::-1])[::-1]
    # class means for all possible thresholds
    # handle divide-by-zero by outputting 0
    mean1 = np.divide(
        np.cumsum(histv * bin_edges),
        weight1,
        np.zeros_like(weight1),
        where=weight1 != 0,
    )
    mean2 = np.divide(
        np.cumsum((histv * bin_edges)[::-1]),
        weight2[::-1],
        out=np.zeros_like(weight2[::-1]),
        where=weight2[::-1] != 0,
    )[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of `weight1`/`mean1` should pair with zero values in
    # `weight2`/`mean2`, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    variance12[np.isnan(variance12)] = 0

    idx = np.argmax(variance12)
    threshold = bin_edges[:-1][idx]
    return float(threshold)


def fast_gaussian(im: np.ndarray, v: float, scale: float = 2) -> np.ndarray:
    ims = cv2.resize(im, (0, 0), fx=1 / scale, fy=1 / scale)
    im2 = cv2.GaussianBlur(ims, (0, 0), v // scale)
    return cv2.resize(im2, (im.shape[1], im.shape[0]), fx=scale, fy=scale)


# def match_templates_masked(image, templates, last=None):
#     if last is not None:
#         pass
#     # mask = cv2.cvtColor()


def otsu_thresh_lb_fraction(image: np.ndarray, fraction: float) -> np.ndarray:
    if len(image.shape) == 3:
        image = np.min(image, axis=2)
    otsu_lb = int(np.mean(image) * fraction)
    tval = otsu_thresh(image, otsu_lb, 255)
    _, thresh = cv2.threshold(image, tval, 255, cv2.THRESH_BINARY)
    return thresh


# eng.traineddata from https://github.com/tesseract-ocr/tessdata/blob/master/eng.traineddata
# noinspection PyArgumentList
tesseract_only = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data"),
    lang="eng",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
tesseract_lstm = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data"),
    oem=tesserocr.OEM.LSTM_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)
# noinspection PyArgumentList
tesseract_lstm_multiline = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data"),
    oem=tesserocr.OEM.LSTM_ONLY,
    psm=tesserocr.PSM.AUTO,
)

# noinspection PyArgumentList
tesseract_futura = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data"),
    lang="Futura",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

lock = Lock()

T = TypeVar("T", int, float, str)
# T = int
@overload
def tesser_ocr(
    image: np.ndarray,
    whitelist: Optional[str] = None,
    invert: bool = False,
    scale: float = 1,
    blur: Optional[float] = None,
    engine: tesserocr.PyTessBaseAPI = None,
    warn_on_fail: bool = True,
) -> str:
    ...


@overload
def tesser_ocr(
    image: np.ndarray,
    expected_type: Callable[[str], T],
    whitelist: Optional[str] = None,
    invert: bool = False,
    scale: float = 1,
    blur: Optional[float] = None,
    engine: tesserocr.PyTessBaseAPI = None,
    warn_on_fail: bool = True,
) -> Optional[T]:
    ...


@no_type_check
def tesser_ocr(
    image: np.ndarray,
    expected_type: Optional[Callable[[str], T]] = None,
    whitelist: Optional[str] = None,
    invert: bool = False,
    scale: float = 1,
    blur: Optional[float] = None,
    engine: tesserocr.PyTessBaseAPI = tesseract_only,
    warn_on_fail: bool = False,
) -> Optional[T]:

    with lock:

        if image.shape[0] <= 1 or image.shape[1] <= 1:
            if not expected_type or expected_type is str:
                return ""
            else:
                return None

        if whitelist is None:
            if expected_type is int:
                whitelist = string.digits
            elif expected_type is float:
                whitelist = string.digits + "."
            else:
                whitelist = string.digits + string.ascii_letters + string.punctuation + " "

        # print('>', whitelist)

        engine.SetVariable("tessedit_char_whitelist", whitelist)
        if invert:
            image = 255 - image
        if scale != 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if blur:
            image = cv2.GaussianBlur(image, (0, 0), blur)

        # if debug:
        #     cv2.imshow('tesser_ocr', image)
        #     cv2.waitKey(0)

        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape
        engine.SetImageBytes(image.tobytes(), width, height, channels, width * channels)
        text: str = engine.GetUTF8Text()
        if " " not in whitelist:
            text = text.replace(" ", "")
        if "\n" not in whitelist:
            text = text.replace("\n", "")

        if not any(c in whitelist for c in string.ascii_lowercase):
            text = text.upper()

        if expected_type:
            try:
                return expected_type(text)
            except Exception as e:
                try:
                    caller = inspect.stack()[1]
                    logger.log(
                        logging.WARNING if warn_on_fail else logging.DEBUG,
                        f"{os.path.basename(caller.filename)}:{caller.lineno} {caller.function} | "
                        f"Got exception interpreting {text!r} as {expected_type.__name__}",
                    )
                except:
                    logger.log(
                        logging.WARNING if warn_on_fail else logging.DEBUG,
                        f"Got exception interpreting {text!r} as {expected_type.__name__}",
                    )
                return None
        else:
            return text


@overload
def tesser_ocr_all(
    images: Sequence[np.ndarray],
    whitelist: Optional[str] = None,
    invert: bool = False,
    scale: float = 1,
    blur: Optional[float] = None,
    engine: tesserocr.PyTessBaseAPI = tesseract_only,
) -> List[str]:
    ...


@overload
def tesser_ocr_all(
    images: Sequence[np.ndarray],
    expected_type: Callable[[str], T],
    whitelist: Optional[str] = None,
    invert: bool = False,
    scale: float = 1,
    blur: Optional[float] = None,
    engine: tesserocr.PyTessBaseAPI = tesseract_only,
) -> List[Optional[T]]:
    ...


@no_type_check
def tesser_ocr_all(
    images: Sequence[np.ndarray],
    expected_type: Optional[Callable[[str], T]] = None,
    whitelist: Optional[str] = None,
    invert: bool = False,
    scale: float = 1,
    blur: Optional[float] = None,
    engine: tesserocr.PyTessBaseAPI = tesseract_only,
) -> List[Optional[T]]:
    return [
        tesser_ocr(
            image,
            expected_type=expected_type,
            whitelist=whitelist,
            invert=invert,
            scale=scale,
            blur=blur,
            engine=engine,
        )
        for image in images
    ]


def otsu_mask(image: np.ndarray, dilate: Optional[int] = 3) -> np.ndarray:
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if dilate:
        mask = cv2.erode(mask, np.ones((2, 2)))
        mask = cv2.dilate(mask, np.ones((dilate, dilate)))
    return cv2.bitwise_and(image, mask)


def unsharp_mask(image: np.ndarray, unsharp: float, weight: float, threshold: Optional[int] = None) -> np.ndarray:
    unsharped = fast_gaussian(image, unsharp, scale=2)
    im = cv2.addWeighted(image, weight, unsharped, 1 - weight, 0)
    if threshold:
        if len(image.shape) == 3:
            gray = np.min(im, axis=2)
        else:
            gray = im
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return thresh
    else:
        return im


def imread(path: str, mode: Optional[int] = None) -> np.ndarray:
    if mode is not None:
        im = cv2.imread(path, mode)
    else:
        im = cv2.imread(path)
    if im is None:
        if os.path.exists(path):
            raise ValueError(f"Unable to read {path} as an image")
        else:
            raise FileNotFoundError(path)
    else:
        return im


# noinspection PyPep8Naming
def findContours(
    image: np.ndarray,
    mode: int,
    method: int,
    contours: Optional[np.ndarray] = None,
    hierarchy: Optional[np.ndarray] = None,
    offset: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r = cv2.findContours(image, mode, method, contours=contours, hierarchy=hierarchy, offset=offset)
    if len(r) == 3:
        return r[1], r[2]
    else:
        return r[0], r[1]


def normalise(
    im: np.ndarray,
    bottom: int = 2,
    top: int = 98,
    min: Optional[int] = None,
    max: Optional[int] = None,
) -> np.ndarray:
    im = im.astype(np.float)
    if min:
        im -= min
    elif bottom:
        im -= np.percentile(im, bottom)
    else:
        im -= np.min(im)
    im = np.clip(im, 0, 255)

    if max:
        im /= max
    elif top:
        im /= np.percentile(im, top)
    else:
        im /= np.max(im)
    return np.clip(im * 255, 0, 255).astype(np.uint8)


class TemplateMatchException(Exception):
    pass


def matchTemplate(
    image: np.ndarray,
    template: np.ndarray,
    method: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        raise TemplateMatchException(
            f"matchTemplate requires template {template.shape} fit within image to match {image.shape}"
        )
    if mask is None:
        return cv2.matchTemplate(image, template, method)
    else:
        if template.shape != mask.shape:
            raise TemplateMatchException(
                f"matchTemplate requires template {template.shape} matches mask {mask.shape}"
            )
        return cv2.matchTemplate(image, template, method, mask=mask)


def match_templates(
    image: np.ndarray,
    templates: Dict[T, Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]],
    method: int,
    template_in_image=True,
    required_match: Optional[float] = None,
    use_masks: bool = False,
    verbose: bool = False,
    previous_match_context: Any = None,
    _previous_match_dict={},
) -> Tuple[float, T]:
    """
    Find the template that best matches `image` OR the first template that meets `required_match`

    :return: A tuple of the match value and key of the best match OR the first match meeting required_match (if provided)
    """
    if not templates:
        raise ValueError("No templates provided (templates was empty dict)")

    previous_match = None
    if previous_match_context is not None:
        previous_match = _previous_match_dict.get(previous_match_context)

    if method in [cv2.TM_SQDIFF_NORMED, cv2.TM_SQDIFF]:
        arrop = np.min
        valop = operator.lt
    else:
        arrop = np.max
        valop = operator.gt

    # TODO: check the last template that matched first
    best: Optional[Tuple[float, T]] = None
    matches = []
    for key, template in sorted(list(templates.items()), key=lambda e: e[0] == previous_match, reverse=True):
        mask = None
        if use_masks:
            template, mask = template
        if template_in_image:
            conv = matchTemplate(image, template, method, mask=mask)
        else:
            conv = matchTemplate(template, image, method, mask=mask)
        match = float(arrop(conv))
        matches.append((key, match))
        if required_match is not None and valop(match, required_match) and not verbose:
            if previous_match_context is not None:
                _previous_match_dict[previous_match_context] = key
            return match, key
        if not best or valop(match, best[0]):
            best = match, key

    if verbose:
        matches.sort(key=lambda e: e[1])
        try:
            import tabulate

            logger.info(f"Got matches:\n{tabulate.tabulate(matches)}")
        except:
            logger.info(f"Got matches:")
            for n, m in matches:
                logger.info(f"    {n}: {m}")

    assert best is not None
    return best


def match_thresh_template(
    image: np.ndarray, template: np.ndarray, threshold: float, match_threshold: float
) -> bool:
    _, thresh_im = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    match = np.max(
        cv2.matchTemplate(
            thresh_im,
            template,
            cv2.TM_CCORR_NORMED,
        )
    )
    return match > match_threshold


# if __name__ == '__main__':
#     print(os.environ['PATH'])
#     print(tesserocr.tesseract_version())
#
#     # noinspection PyArgumentList
#     ocr = tesserocr.PyTessBaseAPI(oem=tesserocr.OEM.LSTM_ONLY)
#
#     gray = image = cv2.imread('C:\\tmp\\gray.png', 0)
#     height, width = image.shape
#     channels = 1
#
#     ocr.SetImageBytes(image.tobytes(), width, height, channels, width * channels)
#     print(ocr.GetUTF8Text())
#
#     ocr.SetImageBytes((255 - image).tobytes(), width, height, channels, width * channels)
#     print(ocr.GetUTF8Text())
#
#     print('--')
#     print(tesser_ocr(gray, whitelist=string.ascii_uppercase))
#     print(tesser_ocr(gray, invert=True, whitelist=string.ascii_uppercase))


def ocr_region(
    frame: Frame,
    regions: ExtractionRegionsCollection,
    region: str,
    engine: tesserocr.PyTessBaseAPI = tesseract_lstm,
    threshold: Optional[int] = 50,
    op=np.min,
    **kwargs,
) -> Optional[str]:
    map_im = regions[region].extract_one(frame.image)
    map_im_gray = 255 - normalise(op(map_im, axis=2), **kwargs)
    # cv2.imshow('map_im_gray', map_im_gray)
    map_text = tesser_ocr(
        map_im_gray,
        engine=engine,
    )
    map_confidence = np.mean(engine.AllWordConfidences())
    logger.debug(f"Got {region}={map_text!r}, confidence={map_confidence}")
    if threshold is not None and map_confidence < threshold:
        logger.warning(
            f"Map confidence for {region}: {map_text!r} below {threshold} (confidence={map_confidence}) - rejecting"
        )
        return None
    return map_text


def bgr_2hsv(colour):
    return cv2.cvtColor(np.array(colour, np.uint8)[np.newaxis, np.newaxis, :], cv2.COLOR_BGR2HSV_FULL)[0, 0]


def hsv2bgr(colour):
    return cv2.cvtColor(np.array(colour, np.uint8)[np.newaxis, np.newaxis, :], cv2.COLOR_HSV2BGR_FULL)[0, 0]


def main() -> None:
    im = cv2.imread("C:/tmp/tesser_ocr.png")
    print(tesser_ocr(im, whitelist=string.digits + ".-", invert=True, scale=4, blur=2))


if __name__ == "__main__":
    main()

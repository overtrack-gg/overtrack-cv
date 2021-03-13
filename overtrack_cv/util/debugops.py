import inspect
from types import GeneratorType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from overtrack_cv.core import imageops

PREV_ARGS = {}


def sliders(
    image: np.ndarray, scale=1, **kwargs: Callable[[Any], Union[np.ndarray, Iterable[np.ndarray]]]
) -> Dict[str, List[int]]:
    assert len(kwargs)
    cv2.namedWindow("sliders")

    updated = False

    def set_updated(*_):
        nonlocal updated
        updated = True

    for fname, function in kwargs.items():
        args = list(inspect.signature(function).parameters)
        assert len(args), "Slider functions must take args"
        for i, arg in enumerate(args[1:]):
            if arg.count("_") == 3:
                name, lv, uv, start = arg.split("_")
            else:
                name, lv, uv = arg.split("_")
                start = lv
            lv, uv, start = int(lv), int(uv), int(start)
            if PREV_ARGS and i < len(PREV_ARGS[fname]):
                pv = PREV_ARGS[fname][i] - lv
            else:
                pv = start
            cv2.createTrackbar(f"{fname}.{name}", "sliders", pv, uv - lv, set_updated)

    while True:
        steps = [image.copy()]
        editable_image = steps[0]
        inargs = {}
        for fname, function in kwargs.items():
            args = list(inspect.signature(function).parameters)
            inargs[fname] = []
            arg: str
            for arg in args[1:]:
                name, lv, uv = arg.split("_")[:3]
                lv, uv = int(lv), int(uv)
                inargs[fname].append(lv + cv2.getTrackbarPos(f"{fname}.{name}", "sliders"))
            PREV_ARGS[fname] = inargs[fname]
            images = function(editable_image, *inargs[fname])
            if isinstance(images, GeneratorType):
                images = list(images)
            if isinstance(images, list) or isinstance(images, tuple):
                editable_image = images[-1]
                steps += list(images)
            else:
                editable_image = images
                steps.append(editable_image)

        height = max(i.shape[0] for i in steps)
        width = max(i.shape[1] for i in steps)

        if height * 1.7 > width:
            stack = np.hstack
        else:
            stack = np.vstack

        steps_v = [
            cv2.copyMakeBorder(s, 0, height - s.shape[0], 0, width - s.shape[1], cv2.BORDER_CONSTANT)
            for s in steps
        ]
        steps_v = [s if len(s.shape) == 3 else cv2.cvtColor(s, cv2.COLOR_GRAY2BGR) for s in steps_v][1:]
        sim = stack(steps_v)
        if scale != 1:
            sim = cv2.resize(
                sim,
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )

        cv2.imshow("sliders", sim)
        cv2.imshow("preview", sim)
        while not updated:
            k = cv2.waitKey(10) & 0xFF
            if k == 27:
                cv2.destroyAllWindows()
                return inargs
        updated = False


def manual_thresh(gray_image: np.ndarray, scale: float = 3.0, _last={}) -> int:
    cv2.namedWindow("thresh")
    cv2.createTrackbar("t", "thresh", _last.get("t", 0), 255, lambda x: None)
    lastt = t = 0
    while True:
        _, thresh = cv2.threshold(gray_image, t, 255, cv2.THRESH_BINARY)
        cv2.imshow(
            "thresh",
            cv2.resize(
                np.vstack((gray_image, thresh)),
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            ),
        )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        t = cv2.getTrackbarPos("t", "thresh")
        _last["t"] = t
        if t != lastt:
            print(t)
            lastt = t
    cv2.destroyAllWindows()
    return t


def manual_thresh_adaptive(gray_image: np.ndarray, scale: float = 3.0) -> Tuple[int, int, int]:
    cv2.namedWindow("thresh")
    cv2.createTrackbar("method", "thresh", 0, 1, lambda x: None)
    cv2.createTrackbar("block_size", "thresh", 0, 100, lambda x: None)
    cv2.createTrackbar("c", "thresh", 100, 200, lambda x: None)
    while True:
        method = [cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.ADAPTIVE_THRESH_MEAN_C][
            cv2.getTrackbarPos("method", "thresh")
        ]
        block_size = cv2.getTrackbarPos("block_size", "thresh") * 2 + 3
        c = cv2.getTrackbarPos("c", "thresh") - 100
        thresh = cv2.adaptiveThreshold(gray_image, 255, method, cv2.THRESH_BINARY, block_size, c)
        cv2.imshow("thresh", cv2.resize(np.vstack((gray_image, thresh)), (0, 0), fx=scale, fy=scale))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
    print(
        ["ADAPTIVE_THRESH_GAUSSIAN_C", "ADAPTIVE_THRESH_MEAN_C"][cv2.getTrackbarPos("method", "thresh")],
        block_size,
        c,
    )
    return method, block_size, c


def manual_thresh_otsu(
    image: np.ndarray,
    template: np.ndarray = None,
    scale: float = 3.0,
    stack: Callable[[Sequence[np.ndarray]], np.ndarray] = np.vstack,
    grayop: Callable[..., Union[np.ndarray, float]] = np.min,
) -> None:
    if len(image.shape) == 3:
        gray_image = grayop(image, axis=2)
    else:
        gray_image = image
    cv2.namedWindow("thresh otsu")
    cv2.createTrackbar("thresh", "thresh otsu", 0, 255, lambda x: None)

    def set_by_mn_mx(*args: Any) -> None:
        mn = cv2.getTrackbarPos("mn", "thresh otsu")
        mx = cv2.getTrackbarPos("mx", "thresh otsu")
        tval = imageops.otsu_thresh(gray_image, mn, mx)
        cv2.setTrackbarPos("thresh", "thresh otsu", int(tval))

    cv2.createTrackbar("mn", "thresh otsu", 0, 255, set_by_mn_mx)
    cv2.createTrackbar("mx", "thresh otsu", 255, 255, set_by_mn_mx)

    fraction_max = 3

    def set_by_fraction(val: int) -> None:
        fraction = val / 100
        otsu_lb = int(np.mean(image) * fraction)
        cv2.setTrackbarPos("mn", "thresh otsu", otsu_lb)
        set_by_mn_mx()

    cv2.createTrackbar("fraction", "thresh otsu", 100, (fraction_max * 100), set_by_fraction)
    set_by_fraction(100)

    old = 0
    while True:
        t = cv2.getTrackbarPos("thresh", "thresh otsu")

        _, thresh = cv2.threshold(gray_image, t, 255, cv2.THRESH_BINARY)
        if t != old:
            old = t
            if template is not None:
                print(np.min(cv2.matchTemplate(thresh, template, cv2.TM_SQDIFF_NORMED)))
        cv2.imshow(
            "thresh otsu",
            cv2.resize(
                stack(
                    (gray_image, thresh, cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1])
                ),
                (0, 0),
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_NEAREST,
            ),
        )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


def manual_unsharp_mask(
    image: np.ndarray, scale: float = 2, inv=False, callback: Optional[Callable[[np.ndarray], None]] = None
) -> Tuple[float, float, float]:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    updated = [True]

    def update(_: int) -> None:
        updated[0] = True

    cv2.namedWindow("unsharp")
    cv2.createTrackbar("size", "unsharp", 40, 100, update)
    cv2.createTrackbar("weight", "unsharp", 40, 500, update)
    cv2.createTrackbar("threshold", "unsharp", 240, 255, update)
    while True:
        size = max(1, cv2.getTrackbarPos("size", "unsharp") / 10)
        weight = cv2.getTrackbarPos("weight", "unsharp") / 10
        threshold = cv2.getTrackbarPos("threshold", "unsharp")

        unsharp = imageops.fast_gaussian(image, size, scale=1)
        im = cv2.addWeighted(image, weight, unsharp, 1 - weight, 0)
        gray = np.min(im, axis=2)
        _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        if threshold == 0:
            thresh = gray
        if inv:
            thresh = 255 - thresh

        cv2.imshow(
            "unsharp",
            cv2.resize(
                np.vstack(
                    (
                        np.hstack(
                            (
                                image,
                                cv2.cvtColor(np.min(image, axis=2), cv2.COLOR_GRAY2BGR),
                                unsharp,
                            )
                        ),
                        np.hstack(
                            (
                                im,
                                cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
                                cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
                            )
                        ),
                    )
                ),
                (0, 0),
                fx=scale,
                fy=scale,
            ),
        )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if updated[0]:
            if callback:
                callback(thresh)

            updated[0] = False

    cv2.destroyAllWindows()
    print(size, weight, threshold)
    return size, weight, threshold


def inrange(image: np.ndarray, scale: float = 2, callback: Optional[Callable[[np.ndarray], None]] = None):
    updated = True

    def update(_: int) -> None:
        nonlocal updated
        updated = True

    cv2.namedWindow("inrange")
    cv2.createTrackbar("b", "inrange", 0, 100, update)

    cv2.createTrackbar("c1_l", "inrange", 0, 255, update)
    cv2.createTrackbar("c1_h", "inrange", 255, 255, update)

    cv2.createTrackbar("c2_l", "inrange", 0, 255, update)
    cv2.createTrackbar("c2_h", "inrange", 255, 255, update)

    cv2.createTrackbar("c3_l", "inrange", 0, 255, update)
    cv2.createTrackbar("c3_h", "inrange", 255, 255, update)

    while True:
        b = cv2.getTrackbarPos("b", "inrange") / 10.0
        c1_l, c2_l, c3_l = [cv2.getTrackbarPos(f"c{i + 1}_l", "inrange") for i in range(3)]
        c1_h, c2_h, c3_h = [cv2.getTrackbarPos(f"c{i + 1}_h", "inrange") for i in range(3)]

        if b:
            im = cv2.GaussianBlur(image, (0, 0), b)
        else:
            im = image

        result = cv2.inRange(im, (c1_l, c2_l, c3_l), (c1_h, c2_h, c3_h))

        cv2.imshow(
            "inrange",
            cv2.resize(
                np.hstack(
                    (
                        image,
                        im,
                        cv2.cvtColor(result, cv2.COLOR_GRAY2BGR),
                    )
                ),
                (0, 0),
                fx=scale,
                fy=scale,
            ),
        )
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        if updated:
            updated = False
            if callback:
                print(callback(result))

    print((c1_l, c2_l, c3_l), (c1_h, c2_h, c3_h))

    return (c1_l, c2_l, c3_l), (c1_h, c2_h, c3_h)


def show_ocr_segmentations(names: List[np.ndarray], **kwargs: Any) -> None:
    from overtrack.overwatch.ocr import big_noodle

    cv2.imshow("names", np.vstack(names))
    segmented_names = []
    for im in names:
        segments = big_noodle.segment(big_noodle.to_gray(im, channel="max"), **kwargs)
        segmented_name = np.hstack(
            cv2.copyMakeBorder(s, 0, 0, 0, 3, cv2.BORDER_CONSTANT, value=128) for s in segments
        )
        segmented_name = cv2.copyMakeBorder(
            segmented_name,
            0,
            5,
            0,
            (names[0].shape[1] + 200) - segmented_name.shape[1],
            cv2.BORDER_CONSTANT,
            value=128,
        )
        segmented_names.append(segmented_name)
    cv2.imshow("segments", np.vstack(segmented_names))
    cv2.waitKey(0)


def tesser_ocr(im: np.ndarray, vscale: float = 3, **kwargs) -> None:
    import overtrack.apex.ocr

    def update(_: int) -> None:
        scale = max(1, cv2.getTrackbarPos("scale", "ocr"))
        if scale < 5:
            scale = 1 / (5 - scale)
        else:
            scale = scale - 3
        blur = max(0, cv2.getTrackbarPos("blur", "ocr") / 10)
        invert = cv2.getTrackbarPos("invert", "ocr")

        print(scale, blur, invert)
        table = []
        if "engine" not in kwargs:
            for name, engine in [
                ("tesseract_lstm", imageops.tesseract_lstm),
                ("tesseract_futura", imageops.tesseract_futura),
                ("tesseract_only", imageops.tesseract_only),
                ("tesseract_ttlakes_digits", overtrack.apex.ocr.tesseract_ttlakes_digits),
                ("tesseract_ttlakes", overtrack.apex.ocr.tesseract_ttlakes),
                ("tesseract_ttlakes_medium", overtrack.apex.ocr.tesseract_ttlakes_medium),
                ("tesseract_arame", overtrack.apex.ocr.tesseract_arame),
                ("tesseract_mensura", overtrack.apex.ocr.tesseract_mensura),
            ]:
                imageops.tesser_ocr(im, scale=scale, blur=blur, invert=bool(invert), engine=engine, **kwargs)
                table.append((name, engine.GetUTF8Text(), engine.AllWordConfidences()))
        else:
            engine = kwargs["engine"]
            imageops.tesser_ocr(
                im,
                scale=scale,
                blur=blur,
                invert=bool(invert),
                engine=engine,
                **{k: v for (k, v) in kwargs.items() if k != "engine"},
            )
            table.append(("", engine.GetUTF8Text(), engine.AllWordConfidences()))
        import tabulate

        print(tabulate.tabulate(table))
        print()

    cv2.namedWindow("ocr")
    cv2.createTrackbar("scale", "ocr", 5, 10, update)
    cv2.createTrackbar("blur", "ocr", 10, 100, update)
    cv2.createTrackbar("invert", "ocr", 0, 1, update)
    while True:
        image = im.copy()
        scale = max(1, cv2.getTrackbarPos("scale", "ocr"))
        if scale < 5:
            scale = 1 / (5 - scale)
        else:
            scale = scale - 3
        blur = max(0, cv2.getTrackbarPos("blur", "ocr") / 10)
        invert = cv2.getTrackbarPos("invert", "ocr")
        if invert:
            image = 255 - image
        if scale != 1:
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        if blur:
            image = cv2.GaussianBlur(image, (0, 0), blur)

        cv2.imshow("image", image)
        cv2.imshow("ocr", cv2.resize(im, (0, 0), fx=vscale, fy=vscale))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()


def test_tesser_engines(image: np.ndarray, scale: float = 1.0) -> None:
    import overtrack.apex.ocr
    import overtrack.valorant.ocr

    if scale != 0:
        image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

    table = []
    for name, engine in [
        ("tesseract_lstm", imageops.tesseract_lstm),
        ("tesseract_futura", imageops.tesseract_futura),
        ("tesseract_only", imageops.tesseract_only),
        ("tesseract_ttlakes_digits", overtrack.apex.ocr.tesseract_ttlakes_digits),
        ("tesseract_ttlakes", overtrack.apex.ocr.tesseract_ttlakes),
        ("tesseract_ttlakes_medium", overtrack.apex.ocr.tesseract_ttlakes_medium),
        ("tesseract_arame", overtrack.apex.ocr.tesseract_arame),
        ("tesseract_mensura", overtrack.apex.ocr.tesseract_mensura),
        ("tesseract_ttlakes_digits_specials", overtrack.apex.ocr.tesseract_ttlakes_digits_specials),
        ("tesseract_ttlakes_bold_digits_specials", overtrack.apex.ocr.tesseract_ttlakes_bold_digits_specials),
        ("din_next_regular_digits", overtrack.valorant.ocr.din_next_regular_digits),
        ("din_next_regular", overtrack.valorant.ocr.din_next_regular),
        ("din_next_combined", overtrack.valorant.ocr.din_next_combined),
    ]:
        if len(image.shape) == 2:
            height, width = image.shape
            channels = 1
        else:
            height, width, channels = image.shape
        engine.SetImageBytes(image.tobytes(), width, height, channels, width * channels)
        table.append((name, engine.GetUTF8Text(), engine.AllWordConfidences()))

    import tabulate

    print(tabulate.tabulate(table))


def manual_canny(gray_image: np.ndarray, scale: float = 3.0) -> int:
    cv2.namedWindow("canny")
    cv2.createTrackbar("t1", "canny", 0, 255, lambda x: None)
    cv2.createTrackbar("t2", "canny", 0, 255, lambda x: None)
    last = None, None
    while True:
        t1, t2 = cv2.getTrackbarPos("t1", "canny"), cv2.getTrackbarPos("t2", "canny")
        out = cv2.Canny(gray_image, t1, t2)
        cv2.imshow("canny", cv2.resize(np.vstack((gray_image, out)), (0, 0), fx=scale, fy=scale))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        # t = cv2.getTrackbarPos('t', 'thresh')
        if (t1, t2) != last:
            last = t1, t2
    cv2.destroyAllWindows()
    return t1, t2


def hstack(images: Sequence[np.ndarray]) -> np.ndarray:
    images = list(images)
    h = max(i.shape[0] for i in images)
    return np.hstack(cv2.copyMakeBorder(i, 0, h - i.shape[0], 0, 0, cv2.BORDER_CONSTANT) for i in images)


def normalise(image, scale=3):
    # image = image.astype(np.float)
    return sliders(
        image,
        normalise=lambda im, bottom_0_100, top_0_100, min_0_255, max_0_255: cv2.resize(
            imageops.normalise(im, bottom_0_100, top_0_100, min_0_255, max_0_255), (0, 0), fx=scale, fy=scale
        ),
    )


def main() -> None:
    # im = cv2.imread("C:/Users/simon/mpv-screenshots/Untitled.png")
    # img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # manual_canny(img)
    # inrange(im)

    im = cv2.imread("C:/tmp/intest.png")[:, :, 1]
    cv2.imshow("input", im)

    sliders(
        im,
        blur=lambda im, sigma_0_50: cv2.GaussianBlur(im, (0, 0), sigma_0_50 / 10) if sigma_0_50 else im,
        filtermax=lambda im, t_0_255, h_1_15, w_1_15: (
            (cv2.dilate(im, np.ones((h_1_15, w_1_15))) == im) & (im > t_0_255)
        ).astype(np.uint8)
        * 255,
    )


if __name__ == "__main__":
    main()

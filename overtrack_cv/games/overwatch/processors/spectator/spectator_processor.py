import base64
import json
import logging
import os
from collections import deque
from pprint import pprint
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

import cv2
import numpy as np
import requests

from overtrack_cv.core import imageops
from overtrack_cv.frame import Frame
from overtrack_cv.games.overwatch.processors.spectator import Player, SpectatorBar
from overtrack_cv.games.processor import Processor

logger = logging.getLogger(__name__)


Colour = Tuple[int, int, int]


def _make_template(bgcol: Colour, hero: str, scale: float) -> np.ndarray:
    p = os.path.join(os.path.dirname(__file__), "data", "spectator_heroes", hero + ".png")
    logger.debug("Loading %s template from %s", hero, p)
    hero_img = imageops.imread(p, -1)
    foreground = hero_img[:, :, :3].astype(np.float)
    alpha = cv2.cvtColor(hero_img[:, :, 3], cv2.COLOR_GRAY2BGR).astype(float) / 255
    # noinspection PyTypeChecker
    background = np.full_like(foreground, bgcol)
    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)
    template = cv2.add(foreground, background).astype(np.uint8)
    template = cv2.resize(template, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return template


class HeroTemplates(NamedTuple):
    hero: str
    blue: np.ndarray
    red: np.ndarray

    def get_by_team(self, is_red_team: bool) -> np.ndarray:
        if is_red_team:
            return self.red
        else:
            return self.blue


def _load_templates(hero: str, scale: float, bgcols: Tuple[Colour, Colour] = None) -> HeroTemplates:
    if bgcols is None:
        bgcols = ((177, 134, 36), (20, 3, 157))
    templates: List[np.ndarray] = []
    for bgcol in bgcols:
        try:
            template = _make_template(bgcol, hero, scale)
        except Exception as e:
            raise ValueError(f"Failed to make template for {hero} x{scale} - {e}")
        templates.append(template)
    return HeroTemplates(hero, templates[0], templates[1])


class SpectatorProcessor(Processor):
    LEFT_X = [50, 1242]

    TOP = 72
    CONTENDERS_TOP = 112
    OWL_TOP = 118

    WIDTH = 106
    HEIGHT = 85
    GAP = 0
    HERO_TEMPLATE_THRESH = 0.12

    NAME_PROCESS_EVERY = 30
    NAME_OFFSET = 40
    NAME_HEIGHT = 22
    NAME_LEFT = [40, 1250]
    NAME_WIDTH = 100
    NAME_GAP = 6

    PORTRAIT_SCALE: Dict[str, float] = {
        "ana": 0.631,
        "baptiste": 0.615,
        "brigitte": 0.560,
        "bastion": 0.681,
        "doomfist": 0.486,
        "dva": 0.55,
        "genji": 0.608,
        "hammond": 0.921,
        "hanzo": 0.7,
        "junkrat": 0.623,
        "lucio": 0.628,
        "mccree": 0.623,
        "mei": 0.603,
        "mercy": 0.603,
        "moira": 0.53,
        "orisa": 0.631,
        "pharah": 0.694,
        "reaper": 0.679,
        "reinhardt": 0.636,
        "roadhog": 0.626,
        "sigma": 0.613,
        "soldier": 0.671,
        "sombra": 0.633,
        "symmetra": 0.628,
        "torbjorn": 0.631,
        "tracer": 0.689,
        "widowmaker": 0.691,
        "winston": 0.734,
        "zarya": 0.621,
        "zenyatta": 0.737,
    }

    def __init__(
        self,
        record_names: bool = False,
        ocr_names: bool = False,
        bgcols: Optional[Tuple[Colour, Colour]] = None,
        top: Optional[int] = None,
    ):
        self.hero_templates: List[HeroTemplates] = self.set_bgcols(bgcols)
        self.set_bgcols(bgcols)
        self.record_name_images = record_names
        self.ocr_names = record_names and ocr_names
        self.top = top or self.TOP
        self.name_images: List[List[np.ndarray]] = [[] for _ in range(12)]
        self.names: List[Optional[str]] = [None for _ in range(12)]
        self.cached_heroes: List[Optional[str]] = [None for _ in range(12)]

        if ocr_names:
            assert "GAPI_VISION_KEY" in os.environ

    def set_bgcols(self, bgcols: Tuple[Colour, Colour]) -> List[HeroTemplates]:
        self.hero_templates = [
            _load_templates(hero, scale, bgcols) for hero, scale in sorted(self.PORTRAIT_SCALE.items())
        ]
        return self.hero_templates

    def _get_hero(self, hero_image: np.ndarray, is_red_team: bool) -> Tuple[float, int]:
        # assert self.hero_templates is not None, 'hero_templates should be initialised from __init__'
        matches = []
        for template in self.hero_templates:
            template = template.get_by_team(is_red_team)
            min_val = np.min(cv2.matchTemplate(template, hero_image, cv2.TM_SQDIFF_NORMED))
            matches.append(min_val)

        hero_match_index = int(np.argmin(matches))
        hero_match = matches[hero_match_index]

        # cv2.imshow('image', hero_image)
        # cv2.imshow('templates', np.hstack([t.get_by_team(is_red_team) for t in self.hero_templates]))
        # print(list(zip([t.hero for t in self.hero_templates], matches)))
        # cv2.waitKey(0)
        return float(hero_match), hero_match_index

    def process(self, frame: Frame) -> bool:
        # assert self.hero_templates is not None, 'hero_templates should be initialised from __init__'
        players: List[Optional[Player]] = [None for _ in range(12)]
        darkened = False
        for is_red_team, x in enumerate(self.LEFT_X):
            for i in range(6):
                ind = is_red_team * 6 + i

                y1 = self.top
                x1 = x + (self.GAP + self.WIDTH) * i

                hero_image = frame.image[y1 + 2 : y1 + 35, x1 + 55 : x1 + 91]
                hero_match, hero_index = self._get_hero(hero_image, bool(is_red_team))
                valid = False
                if hero_match < self.HERO_TEMPLATE_THRESH:
                    self.cached_heroes[ind] = self.hero_templates[hero_index].hero
                    player = Player(
                        self.hero_templates[hero_index][0], match=hero_match, selected=False, name=self.names[ind]
                    )
                    valid = True
                    left = self.NAME_LEFT[is_red_team] + (self.NAME_GAP + self.NAME_WIDTH) * i
                    name_im = frame.image[
                        self.top + self.NAME_OFFSET : self.top + self.NAME_OFFSET + self.NAME_HEIGHT,
                        left : left + self.NAME_WIDTH + 10,
                    ]
                    if self.record_name_images:
                        self.name_images[ind].append(name_im.copy())
                else:
                    player = Player(
                        self.hero_templates[hero_index][0],
                        match=round(hero_match, 2),
                        selected=False,
                        name=self.names[ind],
                    )
                if valid:
                    players[ind] = player
                if player and frame.debug_image is not None:
                    if not darkened:
                        darkened = True
                        dark_region = frame.debug_image[200 + y1 - 20 : 500, 40:800]
                        dark_region[:] = np.clip(dark_region, 100, 255) - 100

                    if player.match:
                        player = player._replace(match=round(player.match, 2))
                    color = (0, 255, 0)
                    if not valid:
                        color = (0, 0, 255)
                    elif not player.match:
                        color = (255, 0, 255)
                    cv2.putText(
                        frame.debug_image,
                        str(player),
                        (50 + i * 40, 200 + y1 + ind * 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        1,
                    )

        if self.ocr_names and max(len(n) for n in self.name_images) >= self.NAME_PROCESS_EVERY:
            # TODO: allow setting this to redo every period
            try:
                self.process_names()
                self.record_name_images = False
            except Exception as e:
                logger.exception("Failed to process spectator names", exc_info=e)
            self.name_images = [deque(maxlen=256) for _ in range(12)]

        if sum(p is not None and p.match is not None for p in players) > 4:
            frame.overwatch.spectator_bar = SpectatorBar(players[:6], players[6:])

            return True
        else:
            return False

    def process_names(self, clear=True) -> None:
        names_images = []
        for i, names in enumerate(self.name_images):
            if len(names):
                meanim = np.mean(names, axis=0).astype(np.uint8)
                names_images.append(meanim)
            else:
                logger.warning(f"Attempted to OCR names when player {i} has 0 images")
                return
                # cv2.imshow('name', np.mean(names, axis=0).astype(np.uint8))
                # cv2.waitKey(0)
        if not len(names_images):
            logger.warning(f"Attempted to OCR 0 name images")
            return
        names = parse_names_google(names_images)
        names_iter = iter(names)
        self.names = [None for _ in range(12)]
        if len(names) < len(names_images):
            logger.error("Only got %d names (%s), expected %d", len(names), names, len(names_images))
        if clear:
            for i, names in enumerate(self.name_images):
                if len(names):
                    # noinspection PyTypeChecker
                    self.names[i] = next(names_iter)
                self.name_images[i].clear()


GAPI_VISION_ENDPOINT_URL = "https://vision.googleapis.com/v1/images:annotate"


def _ocr_images_google(imgs: Sequence[np.ndarray]) -> Optional[Dict[str, Any]]:
    # cv2.imshow('names', np.vstack(imgs))
    # cv2.waitKey(0)

    logger.info("Requesting GAPI OCR...")
    img_requests = []
    for img in imgs:
        # rw = max(1, np.argmax(np.sum(img, axis=(0, 2))[::-1] > 0))
        r, imgdata = cv2.imencode(".png", img)
        assert r
        img_requests.append(
            {
                "image": {"content": base64.b64encode(imgdata).decode()},
                "features": [
                    {
                        "type": "TEXT_DETECTION",
                        "maxResults": 1,
                    }
                ],
                "imageContext": {"languageHints": ["en"]},
            }
        )
    request = json.dumps({"requests": img_requests}).encode()
    response = requests.post(
        GAPI_VISION_ENDPOINT_URL,
        data=request,
        params={"key": os.environ["GAPI_VISION_KEY"]},
        headers={"Content-Type": "application/json"},
    )
    if response.status_code != 200:
        logger.error("Got statuscode %d - %s", response.status_code, response.text)
        return None
    else:
        logger.info("Got statuscode 200")
        return cast(Dict[str, Any], response.json())


def parse_names_google(name_images: Sequence[np.ndarray]) -> List[str]:
    responses = _ocr_images_google(name_images)["responses"]
    pprint(responses)
    names = [r["fullTextAnnotation"]["text"].strip().upper() for r in responses if "fullTextAnnotation" in r]
    names = [n for n in names if len(n) >= 2]
    logger.info(f"Parsed names as {names}")
    # if len(names) != 12:
    #     raise ValueError('Expected 12 names')
    return names


def parse_names_aws(name_images: Sequence[np.ndarray]) -> List[str]:
    import boto3

    rekognition = boto3.client("rekognition", region_name="ap-southeast-2")
    r = rekognition.detect_text(Image={"Bytes": cv2.imencode(".png", np.vstack(name_images))[1].tobytes()})[
        "TextDetections"
    ]
    lines = [d for d in r if d["Type"] == "LINE" if d["DetectedText"].isupper() and len(d["DetectedText"]) >= 2]
    if len(lines) < 12:
        raise ValueError("Expected 12 names")
    while len(lines) > 12:
        worst_index = int(np.argmin([l["Confidence"] for l in lines]))
        logger.info(
            "Got %d names - removing worst (%s c=%d)",
            len(lines),
            lines[worst_index]["DetectedText"],
            lines[worst_index]["Confidence"],
        )
        lines.pop(worst_index)
    return [l["DetectedText"] for l in lines]


# def parse_names(name_images):
#     try:
#         return parse_names_google(name_images)
#     except ValueError as e:
#         logger.warning('Failed to parse names using google OCR', exc_info=e)
#         return parse_names_aws(name_images)


def find_scale(hero: str) -> None:
    from overtrack_cv.core import html2bgr, imageops

    im = cv2.imread(f"C:/Users/simon/overtrack_2/overwatch_images/spectator/{hero}.png")
    frame = Frame.create(im, 0, True)

    is_red_team = True
    i = 0
    hero = hero
    bgcol = ((177, 134, 36), (20, 3, 157))[is_red_team]
    # bgcol = html2bgr('592674')
    # bgcol = html2bgr('5F2080')
    # top = SpectatorProcessor.CONTENDERS_TOP
    top = SpectatorProcessor.TOP

    x = SpectatorProcessor.LEFT_X[is_red_team]
    y1 = top
    x1 = x + (SpectatorProcessor.GAP + SpectatorProcessor.WIDTH) * i
    hero_image = frame.image[y1 + 2 : y1 + 35, x1 + 55 : x1 + 91]
    # cv2.imshow('hero', hero_image)

    x, y = [], []
    from tqdm import tqdm

    for s in tqdm(np.linspace(0.1, 1.5, 200)):
        try:
            t = _make_template(bgcol, hero, s)
            y.append(np.min(cv2.matchTemplate(t, hero_image, cv2.TM_SQDIFF_NORMED)))
            x.append(s)
        except:
            pass

    cv2.imshow("template", t)

    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(x, y)

    plt.figure()
    plt.imshow(hero_image)

    s = x[np.argmin(y)]
    print(s)
    t = _make_template(bgcol, hero, s)
    plt.figure()
    plt.imshow(cv2.matchTemplate(t, hero_image, cv2.TM_SQDIFF_NORMED))

    plt.show()


if __name__ == "__main__":
    # find_scale('sigma')
    from overtrack_cv.util.test_processor import test_processor

    test_processor(
        SpectatorProcessor(
            bgcols=((255, 255, 255), (130, 70, 90)),
            top=SpectatorProcessor.OWL_TOP,
        ),
        "spectator_bar",
    )

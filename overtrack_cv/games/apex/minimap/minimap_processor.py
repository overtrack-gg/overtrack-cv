import logging
import os
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import requests

from overtrack_cv.core import imageops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import CurrentGame, Frame, SerializableArray
from overtrack_cv.games.apex import data as apex_data
from overtrack_cv.games.apex import ocr
from overtrack_cv.games.apex.minimap.models import Location, Minimap, RingsComposite
from overtrack_cv.games.processor import Processor
from overtrack_cv.util import s2ts
from overtrack_cv.util.logging_config import config_logger

logger = logging.getLogger("MinimapProcessor")


class SerializableRingsComposite(SerializableArray):
    def finalize(self):
        array = np.clip(self.array, 0, 30)
        array = np.clip(array, 0, 30)
        array[array < 2] = 0
        return ((array / 30) * 255).astype(np.uint8)


def _draw_map_location(
    debug_image: Optional[np.ndarray],
    minimap: Minimap,
    map_image: np.ndarray,
    offset_x: int,
    offset_y: int,
    map_template: np.ndarray,
    filtered: np.ndarray,
    edges: Optional[np.ndarray],
    rings: Optional[RingsComposite],
) -> None:

    if debug_image is None:
        return

    lines = [f"location={minimap.location}"]

    if len(map_image.shape) == 2:
        out = cv2.cvtColor(map_image[offset_y:-200, offset_x:-200], cv2.COLOR_GRAY2BGR)
    else:
        out = map_image[offset_y:-200, offset_x:-200]
    overlay = np.zeros_like(out)
    cv2.circle(out, minimap.location.coordinates, 6, (0, 255, 255), 4, cv2.LINE_AA)

    if edges is not None and minimap.inner_circle:
        for im in out, overlay:
            cv2.circle(
                im,
                (
                    int(minimap.inner_circle.coordinates[0]),
                    int(minimap.inner_circle.coordinates[1]),
                ),
                int(minimap.inner_circle.r),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                im,
                (
                    int(minimap.inner_circle.coordinates[0]),
                    int(minimap.inner_circle.coordinates[1]),
                ),
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        lines.append(f"inner={minimap.inner_circle}")

    if edges is not None and minimap.outer_circle:
        for im in out, overlay:
            cv2.circle(
                im,
                (
                    int(minimap.outer_circle.coordinates[0]),
                    int(minimap.outer_circle.coordinates[1]),
                ),
                int(minimap.outer_circle.r),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                im,
                (
                    int(minimap.outer_circle.coordinates[0]),
                    int(minimap.outer_circle.coordinates[1]),
                ),
                1,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        lines.append(f"outer={minimap.outer_circle}")

    debug_image[500 : 500 + filtered.shape[0], 300 : 300 + filtered.shape[1]] = filtered
    if edges is not None:
        debug_image[500 : 500 + edges.shape[0], 550 : 550 + edges.shape[1]] = edges.astype(np.uint8) * 255
    debug_image[500 : 500 + filtered.shape[0], 800 : 800 + filtered.shape[1]] = cv2.cvtColor(
        filtered[:, :, 0], cv2.COLOR_GRAY2BGR
    )
    # debug_image[500:500 + filtered.shape[0], 1050:1050 + filtered.shape[1]] = cv2.cvtColor(map_template[
    #     minimap.location.coordinates[1] - 240 // 2 + 8:
    #     minimap.location.coordinates[1] + 240 // 2 - 8,
    #     minimap.location.coordinates[0] - 240 // 2 + 8:
    #     minimap.location.coordinates[0] + 240 // 2 - 8
    # ], cv2.COLOR_GRAY2BGR)

    if minimap.location.bearing is None:
        # draw synthetic minimap. This should match the actual minimap, if it doesn't then something is wrong in the parsing
        debug_image[400 : 400 + 240, 50 : 50 + 240] = out[
            minimap.location.coordinates[1] - 240 // 2 : minimap.location.coordinates[1] + 240 // 2,
            minimap.location.coordinates[0] - 240 // 2 : minimap.location.coordinates[0] + 240 // 2,
        ]
        debug_image[500 : 500 + filtered.shape[0], 1050 : 1050 + filtered.shape[1]] = cv2.cvtColor(
            map_template[
                minimap.location.coordinates[1]
                - filtered.shape[0] // 2 : minimap.location.coordinates[1]
                + filtered.shape[0] // 2,
                minimap.location.coordinates[0]
                - filtered.shape[1] // 2 : minimap.location.coordinates[0]
                + filtered.shape[1] // 2,
            ],
            cv2.COLOR_GRAY2BGR,
        )

        # draw the circle overlay on top of the actual minimap
        if minimap.location.zoom == 1:
            cv2.addWeighted(
                debug_image[50 : 50 + 240, 50 : 50 + 240],
                1,
                overlay[
                    minimap.location.coordinates[1] - 240 // 2 : minimap.location.coordinates[1] + 240 // 2,
                    minimap.location.coordinates[0] - 240 // 2 : minimap.location.coordinates[0] + 240 // 2,
                ],
                0.25,
                0,
                dst=debug_image[50 : 50 + 240, 50 : 50 + 240],
            )

        # draw the minimap rectangle on the full map
        cv2.rectangle(
            out,
            (
                minimap.location.coordinates[0] - int(240 * minimap.location.zoom) // 2,
                minimap.location.coordinates[1] - int(240 * minimap.location.zoom) // 2,
            ),
            (
                minimap.location.coordinates[0] + int(240 * minimap.location.zoom) // 2,
                minimap.location.coordinates[1] + int(240 * minimap.location.zoom) // 2,
            ),
            (0, 255, 255),
            4,
        )
        outtl = (
            minimap.location.coordinates[0] - int(240 * minimap.location.zoom) // 2,
            minimap.location.coordinates[1] - int(240 * minimap.location.zoom) // 2,
        )
    else:
        # rotate the map so we can draw the synthetic minimap with matching rotation
        height, width = out.shape[:2]
        rot = cv2.getRotationMatrix2D(minimap.location.coordinates, minimap.location.bearing - 360, 1)
        rout = cv2.warpAffine(out, rot, (width, height))
        rout = cv2.resize(rout, (0, 0), fx=1 / minimap.location.zoom, fy=1 / minimap.location.zoom)
        debug_image[400 : 400 + 240, 50 : 50 + 240] = rout[
            int(minimap.location.coordinates[1] / minimap.location.zoom)
            - 240 // 2 : int(minimap.location.coordinates[1] / minimap.location.zoom)
            + 240 // 2,
            int(minimap.location.coordinates[0] / minimap.location.zoom)
            - 240 // 2 : int(minimap.location.coordinates[0] / minimap.location.zoom)
            + 240 // 2,
        ]

        # draw the expected template next to the actual filtered output
        trout = cv2.warpAffine(map_template, rot, (width, height))
        trout = cv2.resize(trout, (0, 0), fx=1 / minimap.location.zoom, fy=1 / minimap.location.zoom)
        debug_image[500 : 500 + filtered.shape[0], 1050 : 1050 + filtered.shape[1]] = cv2.cvtColor(
            trout[
                int(minimap.location.coordinates[1] / minimap.location.zoom)
                - 240 // 2
                + 8 : int(minimap.location.coordinates[1] / minimap.location.zoom)
                + 240 // 2
                - 8,
                int(minimap.location.coordinates[0] / minimap.location.zoom)
                - 240 // 2
                + 8 : int(minimap.location.coordinates[0] / minimap.location.zoom)
                + 240 // 2
                - 8,
            ],
            cv2.COLOR_GRAY2BGR,
        )

        # rotate then draw the circles on top of the actual minimap
        roverlay = cv2.warpAffine(overlay, rot, (width, height))
        roverlay = cv2.resize(roverlay, (0, 0), fx=1 / minimap.location.zoom, fy=1 / minimap.location.zoom)
        cv2.addWeighted(
            debug_image[50 : 50 + 240, 50 : 50 + 240],
            1,
            roverlay[
                int(minimap.location.coordinates[1] / minimap.location.zoom)
                - 240 // 2 : int(minimap.location.coordinates[1] / minimap.location.zoom)
                + 240 // 2,
                int(minimap.location.coordinates[0] / minimap.location.zoom)
                - 240 // 2 : int(minimap.location.coordinates[0] / minimap.location.zoom)
                + 240 // 2,
            ],
            0.25,
            0,
            dst=debug_image[50 : 50 + 240, 50 : 50 + 240],
        )

        # work out the corners of the minimap, and draw these on the full map
        tl = (
            minimap.location.coordinates[0] - (240 * minimap.location.zoom) // 2,
            minimap.location.coordinates[1] - (240 * minimap.location.zoom) // 2,
        )
        tr = (
            minimap.location.coordinates[0] + (240 * minimap.location.zoom) // 2,
            minimap.location.coordinates[1] - (240 * minimap.location.zoom) // 2,
        )
        bl = (
            minimap.location.coordinates[0] - (240 * minimap.location.zoom) // 2,
            minimap.location.coordinates[1] + (240 * minimap.location.zoom) // 2,
        )
        br = (
            minimap.location.coordinates[0] + (240 * minimap.location.zoom) // 2,
            minimap.location.coordinates[1] + (240 * minimap.location.zoom) // 2,
        )
        points = cv2.transform(np.array([[tl, tr, br, bl]]), rot).astype(np.int32)
        cv2.polylines(
            out,
            points,
            True,
            (0, 255, 255),
            4,
        )
        outtl = (
            np.min(points[:, :, 0]),
            np.min(points[:, :, 1]),
        )

    if rings:
        for i in rings.images:
            im = np.clip(rings.images[i].array, 0, 30)[:-100, :-100]
            im /= max(np.max(im), 2)
            im = np.stack((im * 200, im * 255, np.zeros_like(im)), axis=-1)
            im = im.astype(np.uint8)
            im = cv2.resize(im, (out.shape[1], out.shape[0]))
            cv2.add(out, im, dst=out)

    out = out[
        outtl[1] - 50 : outtl[1] + 50 + 240 + 50,
        outtl[0] - 50 : outtl[0] + 50 + 240 + 50,
    ]
    debug_image[100 : 100 + out.shape[0], 300 : 300 + out.shape[1]] = out

    for i, line in enumerate(lines):
        cv2.putText(
            debug_image,
            line,
            (730, 250 + 50 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 0),
            3,
        )
        cv2.putText(
            debug_image,
            line,
            (730, 250 + 50 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            1,
        )


class MinimapProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(
        os.path.join(os.path.dirname(__file__), "..", "data", "regions", "16_9.zip")
    )
    SPECTATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "spectate.png"), 0)
    THRESHOLD = 0.1

    offset_x = 0
    offset_y = 0

    @classmethod
    def load_map(cls, path: str):
        im = imageops.imread(path, 0)
        w_h = max(im.shape[0] + 480, im.shape[1] + 480)
        cls.offset_y = (w_h - im.shape[0]) // 2
        cls.offset_x = (w_h - im.shape[1]) // 2
        cls.MAP = cv2.copyMakeBorder(
            im,
            cls.offset_y,
            cls.offset_y,
            cls.offset_x,
            cls.offset_x,
            cv2.BORDER_CONSTANT,
        )
        # cls.MAP_TEMPLATE = cv2.convertScaleAbs(cls.MAP, alpha=1.6, beta=-30)

        LUT = np.linspace(-5, 10, 256)
        LUT = 1 / (1 + np.exp(-LUT * 1.5))
        cls.LUT = (LUT * 255).astype(np.uint8)
        cls.MAP_TEMPLATE = cv2.LUT(cls.MAP, cls.LUT)
        cls.MAP_TEMPLATE = cv2.GaussianBlur(cls.MAP_TEMPLATE, (0, 0), 1.5)  # .astype(np.float)
        # cls.MAP_TEMPLATE *= 1.1
        # cls.MAP_TEMPLATE = np.clip(cls.MAP_TEMPLATE, 0, 255).astype(np.uint8)

        cls.MAP = cls.MAP_TEMPLATE
        # cls.MAP_TEMPLATE = cv2.GaussianBlur(cls.MAP, (0, 0), 1.1).astype(np.float)
        # cls.MAP_TEMPLATE *= 2
        # cls.MAP_TEMPLATE = np.clip(cls.MAP_TEMPLATE, 0, 255).astype(np.uint8)

    MAP_VERSION = 0

    def __init__(self, use_tflite: bool = True):
        self.map_rotated = deque(maxlen=10)
        self.map_rotate_in_config = None
        if use_tflite:
            from overtrack_cv.core.tflite import TFLiteModel

            self.model = TFLiteModel(os.path.join(os.path.dirname(__file__), "data", "minimap_filter.tflite"))
        else:
            from overtrack_cv.core.tf import load_model

            self.model = load_model(
                os.path.join(os.path.dirname(__file__), "data", "minimap_filter")
                # "C:/Users/simon/overtrack_2/training/apex_minimap/v3/v15/checkpoint"
            )
            # from tensorflow.python.keras.saving import export_saved_model
            # export_saved_model(self.model, os.path.join(os.path.dirname(__file__), 'data', 'minimap_filter'), serving_only=True)

        self.current_game: Optional[CurrentGame] = None
        self.current_composite: Optional[RingsComposite] = None

    def eager_load(self):
        self.REGIONS.eager_load()
        self._check_for_rotate_setting()
        self._check_for_map_update()

    def _check_for_rotate_setting(self):
        try:
            # noinspection PyUnresolvedReferences
            from client.util import knownpaths

            games_path = knownpaths.get_path(knownpaths.FOLDERID.SavedGames, knownpaths.UserHandle.current)
            config_path = os.path.join(games_path, "Respawn", "Apex", "profile", "profile.cfg")
            value = None
            with open(config_path) as f:
                for line in f.readlines():
                    if line.startswith("hud_setting_minimapRotate"):
                        value = line.split()[1].strip().replace('"', "")
            if value:
                pvalue = value.lower() in ["1", "true"]
                logger.info(
                    f"Extracted hud_setting_minimapRotate: {value!r} from {config_path} - setting rotate to {pvalue}"
                )
                self.map_rotate_in_config = pvalue
            else:
                logger.info(
                    f"Could not find hud_setting_minimapRotate in {config_path} - setting rotate to autodetect"
                )
                self.map_rotate_in_config = None

        except:
            logger.exception(
                f"Failed to read hud_setting_minimapRotate from profile - setting rotate to autodetect"
            )
            self.map_rotate_in_config = None

    def _check_for_map_update(self):
        logger.info("Checking for map updates")
        try:
            r = requests.get("https://overtrack-client-2.s3-us-west-2.amazonaws.com/dynamic/apex-map/current.json")
            logger.info(f"Checking for map update: {r} {r.content!r}")
            if r.status_code == 404:
                logger.info("Map updates not enabled")
                return

            data = r.json()
            if data["version"] <= self.MAP_VERSION:
                logger.info(
                    f'Current version {self.MAP_VERSION} is up to date - update version is {data["version"]}'
                )
                return
            else:
                maps_path = os.path.join(os.path.join(os.path.dirname(__file__), "data", "maps"))
                os.makedirs(maps_path, exist_ok=True)

                map_path = os.path.join(maps_path, f'{data["version"]}.png')
                if os.path.exists(map_path):
                    try:
                        self.__class__.load_map(map_path)
                    except:
                        logger.info("Map corrupted")
                        os.remove(map_path)
                    else:
                        logger.info(f'Loaded map {data["version"]} from {map_path}')
                        return

                logger.info(f'Downloading map {data["version"]} from {data["url"]} to {map_path}')
                with requests.get(data["url"], stream=True) as r:
                    r.raise_for_status()
                    with open(map_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                self.__class__.load_map(map_path)

        except:
            logger.exception("Failed to check for map update")

    def update(self):
        self._check_for_rotate_setting()
        self._check_for_map_update()

    def process(self, frame: Frame):
        spectate_image = frame.image_yuv[40 : 40 + 30, 670 : 670 + 130, 0]
        _, spectate_image_t = cv2.threshold(spectate_image, 220, 255, cv2.THRESH_BINARY)
        is_spectate = np.max(cv2.matchTemplate(spectate_image_t, self.SPECTATE, cv2.TM_CCORR_NORMED)) > 0.9

        if not is_spectate:
            map_image = frame.image[50 : 50 + 240, 50 : 50 + 240]
        else:
            map_image = frame.image[114 : 114 + 240, 50 : 50 + 240]

        map_image_y = cv2.cvtColor(map_image, cv2.COLOR_BGR2YUV)[:, :, 0]
        map_image_y = cv2.LUT(map_image_y, self.LUT)
        map_image_y = cv2.GaussianBlur(map_image_y, (0, 0), 1.5).astype(np.float)
        map_image_y *= 1.5
        map_image_y = np.clip(map_image_y, 0, 255).astype(np.uint8)

        t0 = time.perf_counter()
        filtered_minimap, filtered_rings = [
            np.clip(p[0], 0, 255).astype(np.uint8)
            for p in self.model.predict(np.expand_dims(map_image, axis=0).astype(np.float32), 1)
        ]
        logger.debug(f"predict {(time.perf_counter() - t0) * 1000:.2f}")

        filtered = np.concatenate(
            (
                np.expand_dims(map_image_y[8:-8, 8:-8], axis=-1),
                cv2.resize(
                    filtered_rings,
                    (filtered_minimap.shape[1], filtered_minimap.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ),
            ),
            axis=-1,
        )

        # location, min_loc, min_val = self._get_location(filtered[:, :, 0])
        location = None
        zoom = self._get_zoom(frame)

        t0 = time.perf_counter()
        if self.map_rotate_in_config or (
            len(self.map_rotated) and (sum(self.map_rotated) / len(self.map_rotated)) > 0.75
        ):
            # 75% of last 10 frames said map was rotated - check rotated first
            logger.debug(f"Checking rotated first")
            bearing = self._get_bearing(frame, frame.debug_image)
            if bearing is not None:
                location = self._get_location(map_image_y, bearing, zoom=zoom)
                logger.debug(f"Got rotated location={location}")
            if (location is None or location.match > self.THRESHOLD) and self.map_rotate_in_config is None:
                # try unrotated
                alt_location = self._get_location(map_image_y, zoom=zoom)
                logger.debug(f"Got unrotated location={alt_location}")
                if location is None or alt_location.match < location.match:
                    location = alt_location
                    bearing = None
        else:
            logger.debug(f"Checking unrotated first")
            location = self._get_location(map_image_y, zoom=zoom)
            logger.debug(f"Got unrotated location={location}")
            if location.match > self.THRESHOLD and self.map_rotate_in_config is None:
                bearing = self._get_bearing(frame, frame.debug_image)
                if bearing is not None:
                    alt_location = self._get_location(map_image_y, bearing, zoom=zoom)
                    logger.debug(f"Got rotated location={alt_location}")
                    if alt_location.match < location.match:
                        location = alt_location
                    else:
                        bearing = None
        logger.debug(f"match {(time.perf_counter() - t0) * 1000:.2f}")

        logger.debug(f"Got location: {location}")
        if location:
            self.map_rotated.append(location.bearing is not None)

            t0 = time.perf_counter()
            self._update_composite(frame, location, filtered_rings)
            logger.debug(f"update composite {(time.perf_counter() - t0) * 1000:.2f}")

            t0 = time.perf_counter()
            blur = cv2.GaussianBlur(filtered, (0, 0), 4)

            blur[:, :, 0] = 0
            edges = self.filter_edge(blur, 50, 20, 20, 10)
            edges[:5, :] = 0
            edges[-5:, :] = 0
            edges[:, :5] = 0
            edges[:, -5:] = 0
            logger.debug(f"filter edges {(time.perf_counter() - t0) * 1000:.2f}")

            t0 = time.perf_counter()
            frame.minimap = Minimap(
                location,
                None,
                None,
                spectate=is_spectate,
                rings_composite=self.current_composite,
                version=3,
            )
            logger.debug(f"get circles {(time.perf_counter() - t0) * 1000:.2f}")

            try:
                _draw_map_location(
                    frame.debug_image,
                    frame.minimap,
                    self.MAP,
                    self.offset_x,
                    self.offset_y,
                    self.MAP_TEMPLATE,
                    filtered,
                    edges,
                    self.current_composite,
                )
            except:
                logger.exception("Failed to draw debug map location")

            return True
        elif location:
            try:
                _draw_map_location(
                    frame.debug_image,
                    Minimap(
                        location,
                        None,
                        None,
                    ),
                    self.MAP,
                    self.MAP_TEMPLATE,
                    filtered,
                    None,
                    self.current_composite,
                )
            except Exception as e:
                pass
                # traceback.print_exc(e)
        return False

    def _get_zoom(self, frame):
        zoom = 1
        if "game_time" in frame:
            if frame.game_time > 1100:
                # round 5 closing / ring 6
                zoom = 0.375
            elif frame.game_time > 980:
                # round 4 closing / ring 5
                zoom = 0.75
        return zoom

    def _get_bearing(self, frame: Frame, debug_image: Optional[np.ndarray]) -> Optional[int]:
        bearing_image = self.REGIONS["bearing"].extract_one(frame.image_yuv[:, :, 0])
        _, bearing_thresh = cv2.threshold(bearing_image, 190, 255, cv2.THRESH_BINARY)

        if debug_image is not None:
            debug_image[
                90 : 90 + bearing_image.shape[0],
                1020 : 1020 + bearing_image.shape[1],
            ] = cv2.cvtColor(bearing_image, cv2.COLOR_GRAY2BGR)
            debug_image[
                90 : 90 + bearing_image.shape[0],
                1100 : 1100 + bearing_image.shape[1],
            ] = cv2.cvtColor(bearing_thresh, cv2.COLOR_GRAY2BGR)

        bearing = imageops.tesser_ocr(
            bearing_thresh,
            expected_type=int,
            engine=ocr.tesseract_ttlakes_digits,
            warn_on_fail=False,
        )
        if bearing is None or not 0 <= bearing <= 360:
            logger.debug(f"Got invalid bearing: {bearing}")
            return None
        if bearing is not None:
            logger.debug(f"Got bearing={bearing}")
            return bearing
        else:
            return None

    def _get_location(
        self,
        region: np.ndarray,
        bearing: Optional[int] = None,
        zoom: Optional[float] = None,
        base_template=None,
    ) -> Location:
        if base_template is None:
            base_template = self.MAP_TEMPLATE

        rot = None
        if bearing is None:
            map_template = base_template
        else:
            height, width = base_template.shape[:2]
            rot = cv2.getRotationMatrix2D(
                (base_template.shape[1] // 2, base_template.shape[0] // 2),
                bearing - 360,
                1,
            )
            map_template = cv2.warpAffine(base_template, rot, (width, height))

        if zoom and zoom != 1:
            region = cv2.resize(region, (0, 0), fx=zoom, fy=zoom)

        # cv2.imshow('map_template', map_template)
        match = cv2.matchTemplate(map_template, region, cv2.TM_SQDIFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match)

        coords = (
            min_loc[0] + int(240 * zoom) // 2 - 8,
            min_loc[1] + int(240 * zoom) // 2 - 8,
        )
        if rot is not None:
            inv = cv2.invertAffineTransform(rot)
            coords = cv2.transform(np.array([[coords]]), inv)[0][0]

        return Location(
            tuple(np.array(coords) - (self.offset_x, self.offset_y)),
            min_val,
            bearing=bearing,
            zoom=zoom,
        )  # , match

    def _update_composite(self, frame: Frame, location: Location, filtered: np.ndarray) -> None:
        current_game: Optional[CurrentGame] = getattr(frame, "current_game", None)
        game_time: Optional[float] = getattr(frame, "game_time", None)

        if current_game:
            if current_game is not self.current_game:
                logger.info(f"Creating new RingsComposite for {frame.current_game}")
                self.current_game = frame.current_game
                self.current_composite = RingsComposite()
        else:
            self.current_game = None
            self.current_composite = None

        if self.current_game and game_time and self.current_composite:
            ring_state = apex_data.get_round_state(game_time)

            to_add = []
            if ring_state.ring_index and not ring_state.ring_closing:
                to_add.append((ring_state.ring_index, filtered[:, :, 1], "outer ring"))
            else:
                logger.debug(
                    f"Not adding outer ring to composite for game time {s2ts(game_time)}, closing={ring_state.ring_closing}"
                )

            if ring_state.next_ring_index:
                to_add.append((ring_state.next_ring_index, filtered[:, :, 0], "inner ring"))
            else:
                logger.debug(f"Not adding inner ring to composite for game time {s2ts(game_time)}")

            for index, image, name in to_add:
                logger.debug(
                    f"Adding {name} to ring composite {index} (approx {np.sum(image > 128)} observed ring pixels) for game time {s2ts(game_time)}"
                )

                # TODO: handle zoom, handle non-rotating
                if location.bearing is None:
                    image_t = image
                else:
                    image = cv2.copyMakeBorder(
                        image,
                        image.shape[0] // 5,
                        image.shape[0] // 5,
                        image.shape[1] // 5,
                        image.shape[1] // 5,
                        cv2.BORDER_CONSTANT,
                    )
                    image_t = cv2.warpAffine(
                        image,
                        cv2.getRotationMatrix2D(
                            (image.shape[1] // 2, image.shape[0] // 2),
                            360 - location.bearing,
                            1,
                        ),
                        (image.shape[0], image.shape[1]),
                    )
                image_t = cv2.resize(image_t, (0, 0), fx=location.zoom, fy=location.zoom)

                if index not in self.current_composite.images:
                    self.current_composite.images[index] = SerializableRingsComposite(
                        np.zeros(
                            (
                                self.MAP_TEMPLATE.shape[0] // 2,
                                self.MAP_TEMPLATE.shape[1] // 2,
                            )
                        )
                    )
                target = self.current_composite.images[index].array

                # TODO: handle borders
                try:
                    y = location.y // 2 - image_t.shape[0] // 2
                    x = location.x // 2 - image_t.shape[1] // 2
                    target[y : y + image_t.shape[0], x : x + image_t.shape[1]] += image_t.astype(np.float) / 255.0
                except:
                    logger.exception("Failed to add ring to composite")

    def filter_edge(
        self,
        im: np.ndarray,
        thresh: int,
        edge_type_box_size: int,
        edge_type_widening: int,
        edge_extraction_size: int,
    ) -> np.ndarray:
        thresh = im > thresh

        x_edge_prominent = cv2.boxFilter(im, 0, (2, edge_type_box_size), normalize=True)
        y_edge_prominent = cv2.boxFilter(im, 0, (edge_type_box_size, 2), normalize=True)
        greater = (x_edge_prominent > y_edge_prominent).astype(np.uint8)
        greater = cv2.dilate(greater, np.ones((1, edge_type_widening)))
        greater = cv2.erode(greater, np.ones((edge_type_widening, 1)))

        w_edge = thresh & (cv2.dilate(im, np.ones((1, edge_extraction_size))) == im)
        h_edge = thresh & (cv2.dilate(im, np.ones((edge_extraction_size, 1))) == im)

        # cv2.imshow('x_edge_prominent', x_edge_prominent)
        # cv2.imshow('y_edge_prominent', y_edge_prominent)
        # cv2.imshow('greater', greater * 255)
        # cv2.imshow('w_edge', w_edge.astype(np.uint8) * 255)
        # cv2.imshow('h_edge', h_edge.astype(np.uint8) * 255)

        edge = (w_edge * greater) + (h_edge * (1 - greater))
        return edge

    # def _get_circle(self, filt: np.ndarray, location: Location) -> Optional[Circle]:
    #     y_i, x_i = np.nonzero(filt)
    #     if len(y_i) > 50:
    #         def calc_R(x, y, xc, yc):
    #             """
    #             calculate the distance of each 2D points from the center (xc, yc)
    #             """
    #             return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    #         def f(c, x, y):
    #             """
    #             calculate the algebraic distance between the data points
    #             and the mean circle centered at c=(xc, yc)
    #             """
    #             Ri = calc_R(x, y, *c)
    #             error = Ri - Ri.mean()
    #             return error
    #
    #         x_m = np.mean(x_i)
    #         y_m = np.mean(y_i)
    #         center_estimate = x_m, y_m
    #         result = optimize.least_squares(
    #             f,
    #             center_estimate,
    #             args=(x_i, y_i),
    #             # loss='soft_l1',
    #         )
    #         center = result.x
    #         x, y = center
    #         r_i = calc_R(x_i, y_i, x, y)
    #         r = r_i.mean()
    #         residu = np.sum((r_i - r) ** 2)
    #
    #         minimap_center_relative = (x - (240 // 2 - 8)), y - (240 // 2 - 8)
    #         if location.bearing is not None:
    #             rot = cv2.getRotationMatrix2D(
    #                 (0, 0),
    #                 360 - location.bearing,
    #                 1
    #             )
    #             minimap_center_relative = tuple(cv2.transform(np.array([[minimap_center_relative]]), rot)[0][0])
    #
    #         return Circle(
    #             (
    #                 location.coordinates[0] + minimap_center_relative[0] * location.zoom,
    #                 location.coordinates[1] + minimap_center_relative[1] * location.zoom
    #             ),
    #             r * location.zoom,
    #             residu,
    #             len(y_i),
    #             )
    #
    #     return None


def main() -> None:
    import glob

    config_logger("map_processor", logging.DEBUG, write_to_file=False)

    # find_best_match()
    # exit(0)

    pipeline = MinimapProcessor()
    paths = glob.glob("d:/overtrack/frames_rot/*.json")[1630:]

    # random.shuffle(paths)

    # paths = ['D:/overtrack/frames_rot/1569668416.76_image.png']
    # paths = [
    #    r'd:/overtrack/frames_rot\1569669144.44_frame.json'
    #     r'd:/overtrack/frames_rot\1569670312.41_image.png'
    # ]

    for p in paths:
        print(p)
        imp = p.replace("frame.json", "image.png")
        try:
            frame = Frame.create(
                cv2.resize(cv2.imread(imp), (1920, 1080)),
                float(os.path.basename(p).split("_")[0]),
                True,
                file=imp,
            )
        except:
            logging.exception("")
            continue
        frame.game_time = frame.timestamp - 1569665630

        pipeline.process(frame)
        print(frame)
        cv2.imshow("debug", frame.debug_image)

        cv2.waitKey(0)


MinimapProcessor.load_map(os.path.join(os.path.dirname(__file__), "data", "9.png"))
MinimapProcessor.MAP_VERSION = 9


def convert_tflite():
    import tensorflow as tf

    p = MinimapProcessor(use_tflite=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(p.model)
    tflite_model = converter.convert()
    with open(os.path.join(os.path.dirname(__file__), "data", "minimap_filter.tflite"), "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    from overtrack_cv.util.test_processor import test_processor

    # convert_tflite()
    # im = cv2.imread('C:/tmp/minimap.png', 0)
    #
    # def lut(im, start_0_20_5, stop_0_20_10, k_0_20_10):
    #     start = start_0_20_5
    #     stop = stop_0_20_10
    #     k = k_0_20_10 / 10
    #
    #     LUT = np.linspace(-start, stop, 256)
    #     LUT = 1 / (1 + np.exp(-LUT * k))
    #     LUT = (LUT * 255).astype(np.uint8)
    #     print(LUT.shape, LUT)
    #
    #     return cv2.LUT(im, LUT)
    # debugops.sliders(
    #     im,
    #     lut=lut,
    # )

    proc = MinimapProcessor()

    # frames = glob.glob("D:/overtrack/frames6/*.json")
    # frames = sorted(frames, key=lambda p: float(os.path.basename(p).split('_')[0]))
    #
    # cg = CurrentGame()
    # for p in tqdm(frames[1600:]):
    #     with open(p) as f:
    #         fdata = json.load(f)
    #     if 'minimap' in fdata:
    #         f = Frame.create(
    #             cv2.imread(p.replace('_frame.json', '_image.png')),
    #             fdata['timestamp'],
    #             debug=True,
    #             current_game=cg,
    #             game_time=fdata.get('game_time')
    #         )
    #         proc.process(f)
    #
    #         if proc.current_composite:
    #             for i in proc.current_composite.images:
    #                 cv2.imshow(f'ring {i}', ((np.clip(proc.current_composite.images[i].array, 0, 30) / 30) * 255).astype(np.uint8))
    #
    #         cv2.imshow('debug', f.debug_image)
    #         cv2.waitKey(0)

    # util.test_processor(proc, 'minimap', 'game_time', 'current_game')
    test_processor(
        "minimap_s3",
        proc,
        "minimap",
        "game_time",
        "current_game",
        game="apex",
        test_all=False,
        warmup=False,
    )

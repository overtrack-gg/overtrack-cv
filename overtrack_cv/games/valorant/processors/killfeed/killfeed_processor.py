import logging
import os
import string
from typing import Dict, NamedTuple, Optional, Tuple

import cv2
import numpy as np

from overtrack_cv.core import imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.frame import Frame
from overtrack_cv.games.processor import Processor
from overtrack_cv.games.valorant.data import AgentName, agents
from overtrack_cv.games.valorant.processors.killfeed.models import (
    Kill,
    Killfeed,
    KillfeedPlayer,
)

logger = logging.getLogger("KillfeedProcessor")


class KillRowPosition(NamedTuple):
    index: int
    match: float
    center: Tuple[int, int]
    friendly: bool


def load_agent_template(name) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(os.path.dirname(__file__), "data", "agents", name.lower() + ".png")
    image1 = imageops.imread(path, -1)[3:-3, 3:-3]

    # path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'agent_icons', name.lower() + '.png')
    # image = imageops.imread(path, -1)
    # image = cv2.resize(image, (33, 33), interpolation=cv2.INTER_NEAREST)[:, ::-1][3:-3, 3:-3]
    # image1 = image

    return image1[:, :, :3], cv2.cvtColor(image1[:, :, 3], cv2.COLOR_GRAY2BGR)


def str2col(s):
    s = sum(ord(c) for c in s) % 255
    return tuple(
        cv2.cvtColor(np.array((s, 230, 255), dtype=np.uint8).reshape((1, 1, 3)), cv2.COLOR_HSV2BGR_FULL)[
            0, 0
        ].tolist()
    )


def draw_weapon_templates(debug_image: Optional[np.ndarray], weapon_templates):
    if debug_image is not None:
        x, y = 300, 100
        for w, t in weapon_templates.items():
            if y > 1000:
                break
            debug_image[y : y + t.shape[0], x : x + t.shape[1]] = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
            cv2.putText(
                debug_image,
                w,
                (
                    x
                    - cv2.getTextSize(w, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 1,)[
                        0
                    ][0],
                    y + 20,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                1,
            )
            y += t.shape[0]


class KillfeedProcessor(Processor):
    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))

    FRIENDLY_KILL_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "friendly_kill.png"), 0
    )
    ENEMY_KILL_TEMPLATE = imageops.imread(os.path.join(os.path.dirname(__file__), "data", "enemy_kill.png"), 0)
    KILL_THRESHOLD = 0.95

    AGENT_DEATH_TEMPLATES: Dict[AgentName, Tuple[np.ndarray, np.ndarray]] = {
        name: load_agent_template(name) for name in agents
    }
    AGENT_KILLER_TEMPLATES: Dict[AgentName, Tuple[np.ndarray, np.ndarray]] = {
        n: (a[0][:, ::-1], a[1][:, ::-1]) for n, a in AGENT_DEATH_TEMPLATES.items()
    }
    AGENT_THRESHOLD = 0.1

    WEAPON_NAMES = [
        "classic",
        "shorty",
        "frenzy",
        "ghost",
        "sheriff",
        "stinger",
        "spectre",
        "bucky",
        "judge",
        "bulldog",
        "guardian",
        "phantom",
        "vandal",
        "marshal",
        "operator",
        "ares",
        "odin",
        "knife",
        "brimstone.incendiary",
        "brimstone.orbital_strike",
        "jett.blade_storm",
        "phoenix.blaze",
        "phoenix.hot_hands",
        "raze.blast_pack",
        "raze.boom_bot",
        "raze.paint_shells",
        "raze.showstopper",
        "sova.hunters_fury",
        "sova.shock_dart",
        "breach.aftershock",
        "viper.snake_bite",
    ]
    WEAPON_IMAGES = {
        n: imageops.imread(os.path.join(os.path.dirname(__file__), "data", "weapons", n + ".png"), 0)
        for n in WEAPON_NAMES
    }
    for n, im in WEAPON_IMAGES.items():
        assert im.shape[1] <= 145, f"{n} image dimensions too high: {im.shape[1]}"
    WEAPON_TEMPLATES = {
        w: cv2.GaussianBlur(
            cv2.dilate(
                cv2.copyMakeBorder(image, 5, 35 - image.shape[0], 5, 145 - image.shape[1], cv2.BORDER_CONSTANT),
                None,
            ),
            (0, 0),
            0.5,
        )
        for w, image in WEAPON_IMAGES.items()
    }
    WEAPON_THRESHOLD = 0.85

    WALLBANG_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "kill_modifiers", "wallbang.png"), 0
    )
    HEADSHOT_TEMPLATE = imageops.imread(
        os.path.join(os.path.dirname(__file__), "data", "kill_modifiers", "headshot.png"), 0
    )
    KILL_MODIFIER_THRESHOLD = 0.75

    def process(self, frame: Frame) -> bool:
        x, y, w, h = self.REGIONS["killfeed"].regions[0]
        region = self.REGIONS["killfeed"].extract_one(frame.image)

        h, s, v = cv2.split(cv2.cvtColor(region, cv2.COLOR_BGR2HSV_FULL))
        h -= 50
        # cv2.imshow('h', h)
        # cv2.imshow('s', s)
        # cv2.imshow('v', v)

        friendly_kill_match = cv2.matchTemplate(h, self.FRIENDLY_KILL_TEMPLATE, cv2.TM_CCORR_NORMED)
        enemy_kill_match = cv2.matchTemplate(h, self.ENEMY_KILL_TEMPLATE, cv2.TM_CCORR_NORMED)
        kill_match = np.max(
            np.stack((friendly_kill_match, enemy_kill_match), axis=-1),
            axis=2,
        )

        kill_rows = []

        for i in range(9):
            mnv, mxv, mnl, mxl = cv2.minMaxLoc(kill_match)
            if mxv < self.KILL_THRESHOLD:
                break

            kill_match[
                max(0, mxl[1] - self.FRIENDLY_KILL_TEMPLATE.shape[0] // 2) : min(
                    mxl[1] + self.FRIENDLY_KILL_TEMPLATE.shape[0] // 2, kill_match.shape[0]
                ),
                max(0, mxl[0] - self.FRIENDLY_KILL_TEMPLATE.shape[1] // 2) : min(
                    mxl[0] + self.FRIENDLY_KILL_TEMPLATE.shape[1] // 2, kill_match.shape[1]
                ),
            ] = 0

            center = (
                int(mxl[0] + x + 20),
                int(mxl[1] + y + self.FRIENDLY_KILL_TEMPLATE.shape[0] // 2),
            )
            friendly_kill_v = friendly_kill_match[mxl[1], mxl[0]]
            enemy_kill_v = enemy_kill_match[mxl[1], mxl[0]]
            logger.debug(
                f"Found kill match at {center}: friendly_kill_v={friendly_kill_v:.4f}, enemy_kill_v={enemy_kill_v:.4f}"
            )

            kill_rows.append(
                KillRowPosition(
                    index=i,
                    match=round(float(mxv), 4),
                    center=center,
                    friendly=bool(friendly_kill_v > enemy_kill_v),
                )
            )

        kill_rows.sort(key=lambda r: r.center[1])
        if len(kill_rows):
            kills = []

            for row in kill_rows:
                killed_agent, killed_agent_match, killed_agent_x = self._parse_agent(frame, row, True)
                if killed_agent_match > self.AGENT_THRESHOLD * 2:
                    continue

                killer_agent, killer_agent_match, killer_agent_x = self._parse_agent(frame, row, False)
                if killer_agent_match > self.AGENT_THRESHOLD * 2:
                    continue

                if killed_agent_match > self.AGENT_THRESHOLD and killer_agent_match > self.AGENT_THRESHOLD:
                    # both invalid - dont bother logging
                    continue
                elif killed_agent_match > self.AGENT_THRESHOLD:
                    logger.warning(
                        f"Ignoring kill {row} - killed_agent_match={killed_agent_match:.1f} ({killed_agent})"
                    )
                    continue
                elif killer_agent_match > self.AGENT_THRESHOLD:
                    logger.warning(
                        f"Ignoring kill {row} - killer_agent_match={killer_agent_match:.1f} ({killer_agent})"
                    )
                    continue

                killed_name = self._parse_killed_name(frame, row, killed_agent_x)
                if killed_name is None:
                    logger.warning(f"Ignoring kill {row} - killed name failed to parse")
                    continue

                weapon, weapon_match, wallbang_match, headshot_match, weapon_x = self._parse_weapon(
                    frame, row, killer_agent_x, killer_agent
                )

                killer_name = self._parse_killer_name(frame, row, killer_agent_x, weapon_x)
                if killer_name is None:
                    logger.warning(f"Ignoring kill {row} - killer name failed to parse")
                    continue

                kill = Kill(
                    y=int(row.center[1]),
                    row_match=round(float(row.match), 4),
                    killer_friendly=row.friendly,
                    killer=KillfeedPlayer(
                        agent=killer_agent,
                        agent_match=round(killer_agent_match, 4),
                        name=killer_name,
                    ),
                    killed=KillfeedPlayer(
                        agent=killed_agent,
                        agent_match=round(killed_agent_match, 4),
                        name=killed_name,
                    ),
                    weapon=weapon,
                    weapon_match=round(weapon_match, 2),
                    wallbang=wallbang_match > self.KILL_MODIFIER_THRESHOLD,
                    wallbang_match=round(wallbang_match, 4),
                    headshot=headshot_match > self.KILL_MODIFIER_THRESHOLD,
                    headshot_match=round(headshot_match, 4),
                )
                kills.append(kill)
                logger.debug(f"Got kill: {kill}")

                if frame.debug_image is not None:
                    s = (
                        f"{row.match:.2f} | "
                        f"{killer_agent} ({killer_agent_match:.4f}) {killer_name!r} >"
                        f' {weapon} {"* " if kill.headshot else ""}{"- " if kill.wallbang else ""}> '
                        f"{killed_agent} ({killed_agent_match:.4f}) {killed_name!r}"
                    )
                    (w, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                    for c, t in ((0, 0, 0), 3), ((0, 255, 128), 1):
                        cv2.putText(
                            frame.debug_image,
                            s,
                            (killer_agent_x - (w + 35), row.center[1] + 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            c,
                            t,
                        )

            if len(kills):
                frame.valorant.killfeed = Killfeed(
                    kills=kills,
                )

                draw_weapon_templates(frame.debug_image, self.WEAPON_TEMPLATES)

                return True

        return False

    _last_image_id = None
    _last_image_names = set()

    def _get_region(self, image, y1, y2, x1, x2, c=None, debug_name=None, debug_image=None):
        if y1 < 0:
            y1 = image.shape[0] + y1
        if y2 < 0:
            y2 = image.shape[0] + y2
        if x1 < 0:
            x1 = image.shape[1] + x1
        if x2 < 0:
            x2 = image.shape[1] + x2
        if debug_image is not None:
            co = str2col(debug_name)
            cv2.rectangle(
                debug_image,
                (x1, y1),
                (x2, y2),
                co,
            )
            if id(debug_image) != self._last_image_id:
                self._last_image_names.clear()
                self._last_image_id = id(debug_image)
            if debug_name and debug_name not in self._last_image_names:
                self._last_image_names.add(debug_name)
                for col, th in ((0, 0, 0), 3), (co, 1):
                    cv2.putText(
                        debug_image,
                        debug_name,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        col,
                        th,
                    )
        region = image[
            y1:y2,
            x1:x2,
        ]
        if c is not None:
            region = region[:, :, c]
        return region

    def _parse_agent(self, frame: Frame, row: KillRowPosition, agent_death: bool) -> Tuple[AgentName, float, int]:
        if agent_death:
            region_x = frame.image.shape[1] - 120
            agent_im = self._get_region(
                frame.image,
                row.center[1] - 20,
                row.center[1] + 20,
                -120,
                -35,
                debug_name="killed_agent",
                debug_image=frame.debug_image,
            )
        else:
            region_x = frame.image.shape[1] - 600
            agent_im = self._get_region(
                frame.image,
                row.center[1] - 20,
                row.center[1] + 20,
                -600,
                row.center[0] - 60,
                debug_name="killer_agent",
                debug_image=frame.debug_image,
            )

        import matplotlib.pyplot as plt

        # cv2.imwrite(f'C:/tmp/agents2/{region_x}.png', agent_im)
        agent_matches = {}
        agent_match_m = []
        t = None
        for a, t in [self.AGENT_KILLER_TEMPLATES, self.AGENT_DEATH_TEMPLATES][agent_death].items():
            match = cv2.matchTemplate(agent_im, t[0], cv2.TM_SQDIFF_NORMED, mask=t[1])
            agent_matches[a] = match
            agent_match_m.append(match)
            # print(a, f'{np.min(match):,}')
            # cv2.imshow(a, t[0])

        import matplotlib.pyplot as plt

        plt.figure()
        plt.imshow(np.vstack(agent_match_m))
        for i, a in enumerate(agents):
            plt.text(-40, int((i + 0.5) * match.shape[0]), a)
            plt.text(
                match.shape[1],
                int((i + 0.5) * match.shape[0]),
                f"{np.min(agent_matches[a]):,}".rjust(12),
                fontdict={"family": "monospace"},
            )
        plt.show()

        agent_match_m = np.min(np.stack(agent_match_m, axis=-1), axis=2)
        mnv, mxv, mnl, mxl = cv2.minMaxLoc(agent_match_m)
        # print(mnv, mnl)

        # print(agent_matches)
        # print(list(zip(self.AGENT_DEATH_TEMPLATES.keys(), agent_match_m)))
        # import matplotlib.pyplot as plt

        # plt.figure()
        # t = self.AGENT_DEATH_TEMPLATES['Breach']
        # plt.imshow(cv2.matchTemplate(agent_im, t[0], cv2.TM_SQDIFF, mask=t[1]))
        # plt.show()

        # plt.figure()
        # plt.imshow(cv2.cvtColor(agent_im, cv2.COLOR_BGR2RGB))
        # plt.figure()
        # plt.imshow(cv2.cvtColor(np.hstack([v[0] for v in self.AGENT_DEATH_TEMPLATES.values()]), cv2.COLOR_BGR2RGB))
        # plt.show()

        agent, agent_match = None, float("inf")
        for a, m in agent_matches.items():
            v = m[mnl[1], mnl[0]]
            if v < agent_match:
                agent_match = v
                agent = a

        return agent, float(agent_match), int(region_x + mnl[0])

    def _parse_killed_name(self, frame, row, killed_agent_x) -> Optional[str]:
        killed_name_gray = self._get_region(
            frame.image_yuv,
            row.center[1] - 10,
            row.center[1] + 10,
            row.center[0] + 10,
            killed_agent_x - 10,
            0,
            debug_name="killed_name",
            debug_image=frame.debug_image,
        )
        if killed_name_gray.shape[1] == 0:
            return None
        killed_name_norm = 255 - imageops.normalise(killed_name_gray, min=170)
        return textops.strip_string(
            imageops.tesser_ocr(killed_name_norm, engine=imageops.tesseract_lstm).upper(),
            alphabet=string.ascii_uppercase + string.digits + "# ",
        )

    def _parse_weapon(
        self, frame, row, killer_agent_x, killer_agent
    ) -> Tuple[Optional[str], float, float, float, int]:
        weapon_region_left = killer_agent_x + 60
        weapon_region_right = row.center[0] - 20
        weapon_gray = self._get_region(
            frame.image_yuv,
            row.center[1] - 15,
            row.center[1] + 17,
            weapon_region_left,
            weapon_region_right,
            0,
            debug_name="weapon",
            debug_image=frame.debug_image,
        )
        if weapon_gray.shape[1] == 0:
            return None, 0, 0, 0, weapon_region_right
        weapon_adapt_thresh = np.clip(
            np.convolve(np.percentile(weapon_gray, 10, axis=0), [0.2, 0.6, 0.2], mode="same"),
            160,
            200,
        )
        weapon_thresh = ((weapon_gray - weapon_adapt_thresh > 30) * 255).astype(np.uint8)

        kill_modifiers_thresh = weapon_thresh[:, -75:]
        _, wallbang_match, _, wallbang_loc = cv2.minMaxLoc(
            cv2.matchTemplate(kill_modifiers_thresh, self.WALLBANG_TEMPLATE, cv2.TM_CCORR_NORMED)
        )
        _, headshot_match, _, headshot_loc = cv2.minMaxLoc(
            cv2.matchTemplate(kill_modifiers_thresh, self.HEADSHOT_TEMPLATE, cv2.TM_CCORR_NORMED)
        )
        wallbang_match, headshot_match = float(wallbang_match), float(headshot_match)
        logger.debug(f"wallbang_match={wallbang_match:.2f}, headshot_match={headshot_match:.2f}")

        right = weapon_thresh.shape[1] - 1
        if wallbang_match > self.KILL_MODIFIER_THRESHOLD:
            right = min(right, (weapon_thresh.shape[1] - 75) + wallbang_loc[0])
        if headshot_match > self.KILL_MODIFIER_THRESHOLD:
            right = min(right, (weapon_thresh.shape[1] - 75) + headshot_loc[0])
        if right != weapon_thresh.shape[1] - 1:
            logger.debug(f"Using right={right} (clipping kill modifier)")
            weapon_thresh = weapon_thresh[:, :right]

        # cv2.imwrite(f'C:/tmp/agents2/weap.png', weapon_thresh)

        # import matplotlib.pyplot as plt
        # f, figs = plt.subplots(4)
        # figs[0].imshow(weapon_gray)
        # figs[1].plot(weapon_adapt_thresh)
        # figs[2].imshow(weapon_gray - weapon_adapt_thresh)
        # figs[3].imshow(weapon_thresh)
        # plt.show()
        # cv2.imshow('weapon_thresh', weapon_thresh)

        weapon_image = cv2.dilate(
            cv2.copyMakeBorder(
                weapon_thresh,
                5,
                5,
                5,
                5,
                cv2.BORDER_CONSTANT,
            ),
            np.ones((2, 2)),
        )
        contours, hierarchy = imageops.findContours(weapon_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_xywh = [(cnt, cv2.boundingRect(cnt)) for cnt in contours]

        best_weap_match, best_weap = 0, None

        for cnt, (x1, y1, w, h) in sorted(contours_xywh, key=lambda cnt_xywh: cnt_xywh[1][0], reverse=True):
            x2, y2 = x1 + w, y1 + h
            a = cv2.contourArea(cnt)

            fromright = weapon_image.shape[1] - x2

            ignore = False
            if w > 145:
                logger.warning(f"Ignoring weapon contour with w={w}")
                ignore = True
            if fromright < 30:
                # contour is far right - could be small agent ability, so be less strict
                if a < 100 or h < 10:
                    logger.debug(
                        f"Ignoring right weapon contour {cv2.boundingRect(cnt)}, fromright={fromright}, a={a}"
                    )
                    ignore = True
                else:
                    logger.debug(
                        f"Allowing potential ability contour {cv2.boundingRect(cnt)}, fromright={fromright}, a={a}"
                    )
            elif a < 200 or h < 16:
                # print('ignore', cv2.boundingRect(cnt), x2, a)
                logger.debug(f"Ignoring weapon contour {cv2.boundingRect(cnt)}, fromright={fromright}, a={a}")
                ignore = True

            if ignore:
                if frame.debug_image is not None and a > 1:
                    cv2.drawContours(
                        frame.debug_image,
                        [cnt],
                        -1,
                        (0, 128, 255),
                        1,
                        offset=(
                            weapon_region_left - 5,
                            row.center[1] - 20,
                        ),
                    )
                continue

            # Draw contour to image, padding l=5, r=10, t=2, b=2
            # The extra width padding prevents abilities matching small parts of large guns
            weapon_im = np.zeros((h + 4, w + 15), dtype=np.uint8)
            cv2.drawContours(
                weapon_im,
                [cnt],
                -1,
                255,
                -1,
                offset=(
                    -x1 + 5,
                    -y1 + 2,
                ),
            )
            if weapon_im.shape[1] > 150:
                weapon_im = weapon_im[:, :150]
            weapon_match, weapon = imageops.match_templates(
                weapon_im,
                {
                    w: t
                    for w, t in self.WEAPON_TEMPLATES.items()
                    if "." not in w or w.lower().startswith(killer_agent.lower() + ".")
                },
                cv2.TM_CCORR_NORMED,
                template_in_image=False,
                required_match=0.96,
                verbose=False,
            )
            if best_weap_match < weapon_match:
                best_weap_match, best_weap = weapon_match, weapon

            valid = weapon_match > self.WEAPON_THRESHOLD

            if frame.debug_image is not None and a > 1:
                cv2.drawContours(
                    frame.debug_image,
                    [cnt],
                    -1,
                    (128, 255, 0) if valid else (0, 0, 255),
                    1,
                    offset=(
                        weapon_region_left - 5,
                        row.center[1] - 20,
                    ),
                )

            if valid:
                if frame.debug_image is not None:
                    x, y = 600, row.center[1] - 15
                    frame.debug_image[
                        y : y + weapon_thresh.shape[0], x : x + weapon_thresh.shape[1]
                    ] = cv2.cvtColor(weapon_thresh, cv2.COLOR_GRAY2BGR)
                    x -= weapon_im.shape[1] + 10
                    frame.debug_image[y : y + weapon_im.shape[0], x : x + weapon_im.shape[1]] = cv2.cvtColor(
                        weapon_im, cv2.COLOR_GRAY2BGR
                    )

                    cv2.line(
                        frame.debug_image,
                        (x, y + weapon_im.shape[0] // 2),
                        (450, self.WEAPON_NAMES.index(weapon) * 40 + 120),
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                return (
                    weapon,
                    float(weapon_match),
                    float(wallbang_match),
                    float(headshot_match),
                    int(weapon_region_left + x1),
                )

        logger.warning(f"Unable to find weapon - best match was {best_weap!r} match={best_weap_match:.2f}")
        return None, 0, 0, 0, weapon_region_right

    def _parse_killer_name(self, frame, row, killer_agent_x, weapon_x) -> Optional[str]:
        killer_name_gray = self._get_region(
            frame.image_yuv,
            row.center[1] - 10,
            row.center[1] + 10,
            killer_agent_x + 35,
            weapon_x - 10,
            0,
            debug_name="killer_name",
            debug_image=frame.debug_image,
        )
        if killer_name_gray.shape[1] == 0:
            return None
        killer_name_norm = 255 - imageops.normalise(killer_name_gray, min=170)
        killer_name = textops.strip_string(
            imageops.tesser_ocr(killer_name_norm, engine=imageops.tesseract_lstm).upper(),
            alphabet=string.ascii_uppercase + string.digits + "#",
        )
        return killer_name


def main():
    from overtrack_cv.util.logging_config import config_logger
    from overtrack_cv.util.test_processor import test_processor

    config_logger(os.path.basename(__file__), level=logging.DEBUG, write_to_file=False)
    proc = KillfeedProcessor()

    test_processor(proc, "valorant.killfeed", wait=True, test_all=False)
    test_processor(proc, "valorant.killfeed", wait=True, test_all=False)


if __name__ == "__main__":
    main()

import glob
import logging
import os
import string
from typing import Dict, Optional, Tuple

import cv2
import Levenshtein as levenshtein
import numpy as np
from overtrack_models.dataclasses.valorant import AgentName

from overtrack_cv.core import imageops, textops
from overtrack_cv.core.region_extraction import ExtractionRegionsCollection
from overtrack_cv.core.uploadable_image import lazy_upload
from overtrack_cv.frame import Frame
from overtrack_cv.games.processor import Processor
from overtrack_cv.games.valorant.data import agents
from overtrack_cv.games.valorant.processors.agent_select.models import AgentSelect

logger = logging.getLogger("AgentSelectProcessor")


def draw_agent_select(debug_image: Optional[np.ndarray], agent_select: AgentSelect) -> None:
    if debug_image is None:
        return

    for c, t in ((0, 0, 0), 4), ((0, 255, 64), 1):
        cv2.putText(
            debug_image,
            str(agent_select),
            (700, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            c,
            t,
        )


def load_rank_template(p: str) -> Tuple[np.ndarray, np.ndarray]:
    image = imageops.imread(p, -1)
    image = cv2.resize(image, (40, 40), cv2.INTER_CUBIC)
    mask = image[:, :, 3]
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    return image[:, :, :3], cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)


class AgentSelectProcessor(Processor):

    REGIONS = ExtractionRegionsCollection(os.path.join(os.path.dirname(__file__), "data", "regions", "16_9.zip"))
    AGENT_NAME_TEMPLATES: Dict[AgentName, np.ndarray] = {
        agent_name: cv2.copyMakeBorder(
            imageops.imread(
                os.path.join(os.path.dirname(__file__), "data", "agent_names", agent_name.lower() + ".png"), 0
            ),
            10,
            10,
            10,
            10,
            cv2.BORDER_CONSTANT,
        )
        #     cv2.resize(
        #     cv2.imread(os.path.join(os.path.dirname(__file__), 'data', 'agent_names', agent_name + '.png'), 0),
        #     (0, 0),
        #     fx=0.5,
        #     fy=0.5,
        # )
        for agent_name in agents
        # if os.path.exists(os.path.join(os.path.dirname(__file__), 'data', 'agent_names', agent_name.lower() + '.png'))
    }
    AGENT_TEMPLATE_REQUIRED_MATCH = 0.95

    RANK_TEMPLATES: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
        str(os.path.basename(p)).rsplit(".", 1)[0]: load_rank_template(p)
        for p in glob.glob(os.path.join(os.path.dirname(__file__), "data", "ranks", "*.png"))
    }

    LOCK_IN_BUTTON_COLOR = (180, 210, 140)

    def __init__(self):
        pass

    def process(self, frame: Frame) -> bool:
        agent_name_yuv = self.REGIONS["agent_name"].extract_one(frame.image_yuv)
        agent_name_thresh = cv2.inRange(agent_name_yuv, (200, 85, 120), (255, 115, 150))
        # if hasattr(frame, 'source_image'):
        # 	cv2.imshow('agent_name_yuv', agent_name_yuv)
        # 	cv2.imshow('agent_name_thresh', agent_name_thresh)
        # 	cv2.imwrite(
        # 		os.path.join(os.path.dirname(__file__), 'data', 'agent_names', os.path.basename(frame.source_image)),
        # 		agent_name_thresh
        # 	)

        match, best_match = imageops.match_templates(
            agent_name_thresh,
            self.AGENT_NAME_TEMPLATES,
            method=cv2.TM_CCORR_NORMED,
            required_match=0.95,
            # verbose=True,
        )
        # self.REGIONS.draw(frame.debug_image)

        if match > self.AGENT_TEMPLATE_REQUIRED_MATCH:
            selected_agent_ims = self.REGIONS["selected_agents"].extract(frame.image)
            selected_agent_ims_gray = [
                255 - imageops.normalise(np.max(im, axis=2), bottom=50) for im in selected_agent_ims
            ]
            selected_agent_texts = imageops.tesser_ocr_all(
                selected_agent_ims_gray,
                engine=imageops.tesseract_lstm,
            )
            logger.info(f"Got selected_agent_texts={selected_agent_texts}")

            picking = True
            for i, text in enumerate(selected_agent_texts):
                for word in textops.strip_string(text, string.ascii_letters + " .").split(" "):
                    match = levenshtein.ratio(word, best_match)
                    logger.debug(f"Player {i}: Got match {match:.2f} for {word!r} = {best_match!r}")
                    if match > 0.7:
                        logger.info(
                            f"Found matching locked in agent {text!r} for selecting agent {best_match!r} - selection locked"
                        )
                        picking = False

            game_mode = imageops.ocr_region(frame, self.REGIONS, "game_mode")

            ranks = []
            for i, im in enumerate(self.REGIONS["player_ranks"].extract(frame.image)):
                match, matched_rank = imageops.match_templates(
                    im,
                    self.RANK_TEMPLATES,
                    method=cv2.TM_SQDIFF,
                    use_masks=True,
                    required_match=15,
                    previous_match_context=("player_ranks", i),
                )
                ranks.append((matched_rank, round(match, 3)))

            player_name_ims = self.REGIONS["player_names"].extract(frame.image)
            player_name_gray = [255 - imageops.normalise(np.max(im, axis=2), bottom=50) for im in player_name_ims]
            player_names = imageops.tesser_ocr_all(player_name_gray, engine=imageops.tesseract_lstm)

            frame.valorant.agent_select = AgentSelect(
                best_match,
                locked_in=not picking,
                map=imageops.ocr_region(frame, self.REGIONS, "map"),
                game_mode=game_mode,
                player_names=player_names,
                agents=selected_agent_texts,
                ranks=ranks,
                image=lazy_upload(
                    "agent_select", self.REGIONS.blank_out(frame.image), frame.timestamp, selection="last"
                ),
            )
            draw_agent_select(frame.debug_image, frame.valorant.agent_select)
            return True

        return False


def main():
    from overtrack_cv.util.logging_config import config_logger
    from overtrack_cv.util.test_processor import test_processor

    config_logger(os.path.basename(__file__), level=logging.DEBUG, write_to_file=False)
    test_processor(AgentSelectProcessor(), "valorant.agent_select", test_all=False)


if __name__ == "__main__":
    main()

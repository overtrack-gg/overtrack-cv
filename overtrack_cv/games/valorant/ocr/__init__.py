import os

import tesserocr

# noinspection PyArgumentList
din_next_regular_digits = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data"),
    lang="din_next_regular_digits",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
din_next_regular = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data"),
    lang="din_next_regular",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
din_next_combined = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data"),
    lang="din_next_combined",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

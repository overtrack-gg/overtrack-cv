import os

import tesserocr

# noinspection PyArgumentList
tesseract_mensura = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data", "fonts"),
    lang="Mensura",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
tesseract_arame = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data", "fonts"),
    lang="Arame",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
tesseract_ttlakes_digits = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data", "fonts"),
    lang="TT_digits",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
tesseract_ttlakes = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data", "fonts"),
    lang="TTLakes_Bold",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
tesseract_ttlakes_medium = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data", "fonts"),
    lang="TTLakes_Medium",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
tesseract_ttlakes_digits_specials = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data", "fonts"),
    lang="TTLakes_digits_specials",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

# noinspection PyArgumentList
tesseract_ttlakes_bold_digits_specials = tesserocr.PyTessBaseAPI(
    path=os.path.join(os.path.dirname(__file__), "data", "fonts"),
    lang="TTLakes_bold_digits_specials",
    oem=tesserocr.OEM.TESSERACT_ONLY,
    psm=tesserocr.PSM.SINGLE_LINE,
)

"""Geometry, colors and font roles of the one-pager template.

Every coordinate is in a 1920x1080 basis (the PowerPoint slide is 13.33 in
wide, so 1 pt = 2 px at 1920). The renderer multiplies everything by
``width / 1920``, so the same numbers serve a 1920, 2560 or 3840 output.

Values were measured from the master slide of the AI4U one-pager deck
(slide "AI-Powered Claims Validation") and from pixels of
``assets/base_plate.png``. If the background template ever changes, these are
the numbers to recalibrate; nothing else in the scripts hardcodes positions.

Inputs: none (constants only).
Outputs: constants imported by ``render_one_pager.py``.
"""

# Template base width. All boxes below are expressed in this basis.
BASE_WIDTH = 1920

# Colors taken from the deck: white body text and the magenta accent used for
# the subtitle, persona roles and the leading verb of each benefit.
WHITE = (255, 255, 255)
ACCENT = (204, 0, 221)  # #CC00DD

# Font roles, mapped to file-name prefixes of the SAP 72 font family.
# render_one_pager.find_font() searches the system font folders for them.
FONTS = {
    "brand": ["72Brand-Regular", "72-Regular"],
    "brand_bold": ["72Brand-Bold", "72-Bold"],
    "regular": ["72-Regular", "72Brand-Regular"],
    "bold": ["72-Bold", "72Brand-Bold"],
}

# Text slots. box = (x, y, w, h). size/min_size in px at 1920 (pt * 2).
# line = line height as a multiple of the font size. valign: "top"/"middle".
# Short single-line slots (title, names) shrink to fit their width instead of
# wrapping, because a wrapped title would collide with the subtitle.
SLOTS = {
    "title": dict(box=(160, 32, 1600, 58), font="brand", size=48, min_size=34,
                  color=WHITE, single_line=True, valign="middle"),
    "subtitle": dict(box=(160, 108, 1600, 40), font="brand_bold", size=32, min_size=24,
                     color=ACCENT, single_line=True, valign="middle"),
    "app_name": dict(box=(640, 262, 640, 44), font="brand", size=32, min_size=22,
                     color=WHITE, single_line=True, valign="middle"),
    "left_name": dict(box=(132, 258, 400, 42), font="bold", size=32, min_size=24,
                      color=WHITE, single_line=True, valign="middle"),
    "left_role": dict(box=(102, 296, 460, 34), font="regular", size=21, min_size=16,
                      color=ACCENT, single_line=True, valign="middle"),
    "right_name": dict(box=(1390, 258, 400, 42), font="bold", size=32, min_size=24,
                       color=WHITE, single_line=True, valign="middle"),
    "right_role": dict(box=(1360, 296, 460, 34), font="regular", size=21, min_size=16,
                       color=ACCENT, single_line=True, valign="middle"),
    # Captions under the arrows, between the persona circles and the laptop.
    "left_caption": dict(box=(446, 462, 296, 150), font="regular", size=21, min_size=16,
                         color=WHITE, line=1.25, valign="top"),
    "right_caption": dict(box=(1172, 462, 296, 150), font="regular", size=21, min_size=16,
                          color=WHITE, line=1.25, valign="top"),
    # The three panels drawn on the plate span y 704-1037; text is centered in
    # an inset box so it never touches the rounded edges.
    "challenge": dict(box=(66, 722, 530, 298), font="regular", size=22, min_size=16,
                      color=WHITE, line=1.75, valign="middle"),
    "solution": dict(box=(695, 722, 530, 298), font="regular", size=22, min_size=16,
                     color=WHITE, line=1.75, valign="middle"),
    "benefits": dict(box=(1324, 722, 530, 298), font="regular", size=22, min_size=16,
                     color=WHITE, line=1.75, valign="middle", para_gap=0.15),
}

# Laptop screen glass on the plate (the screenshot is cover-cropped into it).
SCREEN = (784.5, 319.5, 351.0, 222.0)

# Persona circles: the dark circular backplates drawn on the plate.
LEFT_CIRCLE = (222, 332, 217, 217)
RIGHT_CIRCLE = (1481, 332, 217, 217)

# Smallest output width accepted (1080p) and the default (4K, the native
# resolution of the base plate).
MIN_WIDTH = 1920
DEFAULT_WIDTH = 3840

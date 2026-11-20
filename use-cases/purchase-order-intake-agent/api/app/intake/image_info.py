"""Read image pixel dimensions from PNG, GIF and JPEG headers (standard library only).

Used to tell logos/icons apart from content images (e.g. a screenshot of a purchase order
pasted into an e-mail body) without decoding the image.

Example::

    image_size(open("logo.png", "rb").read())   # (88, 31)
"""

from __future__ import annotations

import struct

# JPEG start-of-frame markers that carry the image size (all SOFn except DHT/JPG/DAC).
_JPEG_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def image_size(data: bytes) -> tuple[int, int] | None:
    """Return (width, height) of a PNG, GIF or JPEG image.

    Args:
        data: Raw image bytes.

    Returns:
        (width, height) in pixels, or None for unknown formats or truncated headers.
    """
    if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
        return struct.unpack("<HH", data[6:10])
    if data[:2] == b"\xff\xd8":
        return _jpeg_size(data)
    return None


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    """Walk JPEG segments until a start-of-frame marker and read its size.

    Args:
        data: JPEG bytes starting with SOI.

    Returns:
        (width, height) or None.
    """
    pos = 2
    while pos + 9 < len(data):
        if data[pos] != 0xFF:
            return None
        marker = data[pos + 1]
        if marker == 0xFF:          # fill byte
            pos += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:   # markers without length
            pos += 2
            continue
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if marker in _JPEG_SOF:
            height, width = struct.unpack(">HH", data[pos + 5:pos + 9])
            return width, height
        pos += 2 + length
    return None

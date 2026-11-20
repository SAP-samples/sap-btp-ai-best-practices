"""Source-intake layer for the ECN impact pipeline.

This package validates uploaded file metadata, enforces bounded byte reading,
parses and normalises ECN JSON arrays, and normalises source objects found
within the parsed ECN records.

Modules:
    files: Upload metadata validation (filename, media type) and bounded read.
    json_parser: UTF-8 ECN JSON decoding, field-fit helpers, ECN normalisation.
    normalization: Source-object normalisation with WTPart identity logic.
"""

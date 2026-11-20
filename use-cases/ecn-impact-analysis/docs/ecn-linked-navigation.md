# Linked ECN navigation

The Overview's **Linked ECN** field reads outgoing references from the current
ECN's source descriptions. If ECN67890 says “linked to ECN12345,” its button
opens the stored ECN12345 occurrence. This does not add a reverse link to
ECN12345. With no outgoing reference, the field shows a disabled **No linked
ECN** button. References to ECNs that have not been imported show a disabled
button bearing the referenced number.

`GET /api/ecns/{ecn_row_id}` returns `linked_ecns`, an ordered list of
`{ecn_number, ecn_row_id}`. A null `ecn_row_id` means the target is unavailable.
Resolution happens on reads, so existing imports work immediately and missing
references become available once their targets are imported. No migration,
summary generation, or model call is required.

The regex scans `description_text` and `description_summary`. Intake preserves
the source `DescriptionSummary`, `Description`, and `SOLUTION_DESCRIPTION` in
`description_text`. Matching ignores case and accepts whitespace (including
tabs/newlines), an optional plural `ECNs`, `No.` / `number`, and separators
such as hyphens, Unicode dashes, underscore, colon, and `#`. Examples include
`ECN12345`, `ecn 12345`, `ECN-12345`, `ECNs: 12345`, and `ECN No. 12345`.

References are deduplicated and self-references excluded. Leading zeroes remain
significant. Embedded words/part IDs, alphanumeric suffixes, bare numbers, and
decimal/comma-formatted numbers do not produce partial matches. Each separate
reference must include an ECN prefix; the parser does not infer a relationship
from unqualified numbers or generated summaries.

Target selection checks all stored occurrences, independent of inbox filters
and pagination, using a single metadata query when references exist. It prefers
an occurrence in the current upload, then the most recently published upload,
then source ordinal and row UUID as stable tie-breakers. The shared Python
identifier regex verifies SQL candidates before enabling navigation. Navigation
preserves the originating inbox search, sort, and page for **Back to ECNs**.

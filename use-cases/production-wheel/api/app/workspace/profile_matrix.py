"""In-memory Excel exchange and validation of complete volume compatibility matrices.

Planner descriptions of a matrix are compiled in ``matrix_description``.

Axis labels are package volumes in litres. Profiles in ``nearest_class`` volume
matching also accept the tail classes ``"<n"`` and ``">n"`` (see
``production_wheel.volume_classes``); exact profiles accept numbers only.
"""

from io import BytesIO
from typing import Literal
from zipfile import BadZipFile, ZipFile

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation
from pydantic import BaseModel, ConfigDict, model_validator
from production_wheel.volume_classes import check_labels, label_sort_key, parse_label

MAX_WORKBOOK_BYTES = 2_000_000


class ProfileMatrixPair(BaseModel):
    """One profile matrix cell: two axis labels (volumes or tail classes) and a status.

    Labels are normalized (``"2.50"`` -> ``"2.5"``, ``"< 1"`` -> ``"<1"``). Whether
    tails are allowed depends on the profile's volume matching and is checked by
    ``validate_matrix_rows``. The diagonal must be ``Y``.
    """

    model_config = ConfigDict(extra='forbid', frozen=True)

    volume_a: str
    volume_b: str
    status: Literal['Y', 'AVOID', 'N']

    @model_validator(mode='before')
    @classmethod
    def normalize(cls, data):
        """Normalize both labels before field validation."""
        if isinstance(data, dict):
            data = {**data, **{k: parse_label(data[k]) for k in ('volume_a', 'volume_b') if k in data}}
        return data

    @model_validator(mode='after')
    def compatible_diagonal(self):
        """Require ``Y`` on the diagonal."""
        if self.volume_a == self.volume_b and self.status != 'Y':
            raise ValueError('matrix diagonal must be compatible')
        return self


def volume_ids(values, allow_classes=True):
    """Normalize axis labels, rejecting duplicates and oversized axes.

    Args:
        values: Labels in axis order.
        allow_classes: Accept ``"<n"`` / ``">n"`` tails.

    Returns:
        Normalized labels in the given order.
    """
    result = [parse_label(v, allow_classes) for v in values]
    if not 1 <= len(result) <= 100 or len(set(result)) != len(result):
        raise ValueError('Use 1–100 unique positive volumes in litres')
    return result


def validate_matrix_rows(rows, volume_matching='nearest_class'):
    """Return a complete symmetric matrix with normalized axes and compatible diagonals.

    Args:
        rows: ``{volume_a, volume_b, status}`` cells (both halves).
        volume_matching: ``exact`` (numbers only) or ``nearest_class`` (tails
            allowed; ``"<n"`` must name the smallest stop, ``">n"`` the largest).

    Returns:
        Normalized rows as dicts, in the given order.
    """
    allow = volume_matching != 'exact'
    values = [ProfileMatrixPair.model_validate(row if isinstance(row, dict) else row.model_dump()) for row in rows]
    pairs = {(r.volume_a, r.volume_b): r.status for r in values}
    labels = {v for pair in pairs for v in pair}
    volumes = volume_ids(sorted(labels, key=label_sort_key), allow)
    if allow:
        check_labels(volumes)
    if len(pairs) != len(values) or any((a, b) not in pairs or pairs[a, b] != pairs.get((b, a))
                                      for a in volumes for b in volumes):
        raise ValueError('Volume matrix must be complete, unique and symmetric; fill both halves')
    return [r.model_dump() for r in values]


def _axis_value(label):
    """Excel header value: a number for a volume, text for a ``<n`` / ``>n`` class."""
    return label if label[:1] in '<>' else float(label)


def matrix_workbook(rows):
    """Generate the editable Excel template from accepted profile rows using the app's Excel library."""
    rows = validate_matrix_rows(rows)
    volumes = sorted({r['volume_a'] for r in rows}, key=label_sort_key)
    pairs = {(r['volume_a'], r['volume_b']): r['status'] for r in rows}
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'Volume compatibility'
    sheet.append(['Volume (litres)', *[_axis_value(v) for v in volumes]])
    for a in volumes:
        sheet.append([_axis_value(a), *[pairs[a, b] for b in volumes]])
    sheet.freeze_panes = 'B2'
    sheet.column_dimensions['A'].width = 20
    for column in range(2, len(volumes) + 2):
        sheet.column_dimensions[get_column_letter(column)].width = 10
    for row in sheet:
        for cell in row:
            cell.alignment = Alignment(horizontal='center')
            cell.font = Font(name='Aptos', size=11)
            if cell.row == 1 or cell.column == 1:
                cell.fill = PatternFill('solid', fgColor='16324F')
                cell.font = Font(name='Aptos', size=11, bold=True, color='FFFFFF')
    choices = DataValidation(type='list', formula1='"Y,AVOID,N"', allow_blank=False)
    choices.errorTitle = 'Choose a compatibility status'
    choices.error = 'Use Y, AVOID or N. Match the opposite cell; diagonal cells must be Y.'
    choices.showErrorMessage = True
    sheet.add_data_validation(choices)
    choices.add('B2:CW101')
    notes = workbook.create_sheet('Instructions')
    for text in (
        'Volume compatibility template',
        'Edit the Volume compatibility sheet. Volumes are positive numbers in litres.',
        'Nearest class profiles: rows are reference volumes plus optional <n (below the smallest) and >n (above the largest) classes; each product takes the row of its nearest reference volume.',
        'Y = compatible. AVOID = discouraged (blocked in HARD mode). N = forbidden in every mode.',
        'Fill every pair using Y, AVOID or N. Both halves must match; diagonal cells must be Y.',
        'To add or remove volumes, edit BOTH the header row and the matching first-column rows. Maximum 100 volumes.',
        'Keep the same volume order on both axes. Do not use formulas or merged cells in the matrix.',
        'Upload the edited .xlsx, review the preview and click Accept. Save profile then persists it.',
        'Accept replaces the whole matrix. Source products with volumes outside it will require a compatible profile.',
    ):
        notes.append([text])
    notes.column_dimensions['A'].width = 105
    for row in notes:
        row[0].font = Font(name='Aptos', size=11)
        row[0].alignment = Alignment(wrap_text=True, vertical='top')
        notes.row_dimensions[row[0].row].height = 32
    output = BytesIO()
    workbook.save(output)
    return output.getvalue()


def read_matrix_workbook(content):
    """Validate a bounded uploaded square matrix; formulas and partial matrices never pass."""
    if len(content) > MAX_WORKBOOK_BYTES:
        raise ValueError('Compatibility workbook exceeds 2 MB')
    try:
        with ZipFile(BytesIO(content)) as archive:
            if len(archive.infolist()) > 100 or sum(i.file_size for i in archive.infolist()) > 10_000_000:
                raise ValueError('Compatibility workbook expands beyond its size limit')
        workbook = load_workbook(BytesIO(content), read_only=True, data_only=False, keep_links=False)
        try:
            if 'Volume compatibility' not in workbook.sheetnames:
                raise ValueError('Use the template sheet named Volume compatibility')
            sheet = workbook['Volume compatibility']
            if sheet.max_row > 101 or sheet.max_column > 101:
                raise ValueError('Matrix supports at most 100 volumes; remove unused rows and columns')
            grid = list(sheet.values)
            while grid and all(value is None for value in grid[-1]):
                grid.pop()
            if not grid or len(grid) < 2:
                raise ValueError('Matrix is empty')
            width = max(i + 1 for row in grid for i, value in enumerate(row) if value is not None)
            grid = [row[:width] for row in grid]
            columns = volume_ids(grid[0][1:])
            rows = volume_ids([row[0] for row in grid[1:]])
            if rows != columns:
                raise ValueError('Row and column volumes must match in the same order')
            values = [{'volume_a': a, 'volume_b': b, 'status': str(grid[i][j] or '').strip().upper()}
                      for i, a in enumerate(rows, 1) for j, b in enumerate(columns, 1)]
            return validate_matrix_rows(values)
        finally:
            workbook.close()
    except (BadZipFile, KeyError, IndexError, TypeError, SyntaxError, EOFError) as exc:
        raise ValueError('Invalid compatibility .xlsx; download and use the template') from exc

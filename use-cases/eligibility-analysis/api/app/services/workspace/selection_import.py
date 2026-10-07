"""Read already eligible extraction uploads without running eligibility rules."""
from hashlib import sha256
from io import BytesIO
from decimal import Decimal, InvalidOperation

import pandas as pd
from ...models.workspace import WorkspaceValidationError


def selection_rows(content):
    """Return source-traceable candidates and import counts; reject malformed candidate rows.

    Historical rows explicitly marked by the upload are excluded. Outcome fields
    are never carried into predictions. Unmarked uploads contain candidates only.
    """
    with pd.ExcelFile(BytesIO(content), engine='openpyxl') as workbook:
        sheet = 'SAPUI5 Export' if 'SAPUI5 Export' in workbook.sheet_names else workbook.sheet_names[0]
        frame = pd.read_excel(workbook, sheet_name=sheet, dtype=object)
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {'Company Code', 'Customer', 'Invoice Reference', 'Offer File Date (UTC)',
                'Due Date', 'Purchase Price', 'Currency', 'Issuance Date'}
    if 'Issuance date' in frame and 'Issuance Date' not in frame:
        frame = frame.rename(columns={'Issuance date': 'Issuance Date'})
    if missing := required - set(frame.columns):
        raise WorkspaceValidationError('Invoice selection requires: '+', '.join(sorted(missing)), 'file')
    rows, historical = [], 0
    source_hash = sha256(content).hexdigest()
    for index, record in enumerate(frame.to_dict('records'), 2):
        if all(pd.isna(value) for value in record.values()):
            continue
        kind = str(record.get('Synthetic Row Type', 'candidate')).strip().lower()
        if kind == 'history':
            historical += 1
            continue
        if kind != 'candidate':
            raise WorkspaceValidationError(f'Row {index}: unknown candidate/history classification', 'file')
        try:
            values = {}
            for key in required:
                value = record[key]
                if pd.isna(value) or not str(value).strip():
                    raise ValueError(f'{key} is required')
                values[key] = str(value).strip()
            for key in ('Issuance Date', 'Due Date', 'Offer File Date (UTC)'):
                timestamp = pd.Timestamp(values[key])
                if pd.isna(timestamp):
                    raise ValueError(f'{key} must be a valid date')
                values[key] = timestamp.isoformat()
            amount = Decimal(values['Purchase Price'])
            if not amount.is_finite():
                raise ValueError('Purchase Price must be finite')
            currency = values['Currency'].upper()
            if len(currency) != 3 or not currency.isalpha():
                raise ValueError('Currency must be a three-letter code')
            funding_currency = record.get('Funding Currency')
            if pd.notna(funding_currency) and str(funding_currency).strip().upper() != currency:
                raise ValueError('Purchase Price currency differs from invoice currency; supply a single-currency extraction')
            face = Decimal(str(record['Amount'])) if pd.notna(record.get('Amount')) else amount
            if not face.is_finite():
                raise ValueError('Amount must be finite')
            invoice = dict(seller_id=values['Company Code'], seller_name=values['Company Code'],
                debtor_id=values['Customer'], debtor_name=values['Customer'],
                invoice_ref=values['Invoice Reference'], original_currency=currency,
                amount_original=str(face), total_net_value_original=str(amount),
                issuance_date=values['Issuance Date'], due_date=values['Due Date'],
                offer_date=values['Offer File Date (UTC)'], programa='', insurer_id='')
            for source, target in [('PROGRAMA', 'programa'), ('Document Number', 'doc_number'),
                                   ('Fiscal Year', 'fiscal_year')]:
                if pd.notna(record.get(source)):
                    invoice[target] = str(record[source]).strip()
            identity = sha256(f'{source_hash}:{sheet}:{index}'.encode()).hexdigest()
            rows.append(dict(row_id=identity, source_event_id=identity, source_row_number=index,
                source_sheet=sheet, eligible=True, eligibility_source='upstream', invoice=invoice,
                diagnostics={'failed_rules': [], 'eligibility_source': 'upstream'}))
        except (ValueError, TypeError, InvalidOperation) as error:
            raise WorkspaceValidationError(f'Row {index}: {error}', 'file') from None
    if not rows:
        raise WorkspaceValidationError('No candidate invoices found in the extraction', 'file')
    return rows, {'source_kind': 'selection', 'historical_rows_excluded': historical,
                  'eligibility_source': 'upstream'}

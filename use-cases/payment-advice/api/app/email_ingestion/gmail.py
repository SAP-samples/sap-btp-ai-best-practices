"""Read-only Gmail through the dedicated SAP Destination; never expose tokens."""
import base64
import json
import os
import time
from datetime import datetime, timezone
from email.utils import parseaddr
from html.parser import HTMLParser
import requests
from .domain import MAX_ATTACHMENTS, MAX_FILE, MAX_TOTAL

SERVICE = 'payment-advice-automation-destination'
DESTINATION = 'PAYMENT_ADVICE_GMAIL_READONLY'
BASE = 'https://gmail.googleapis.com/gmail/v1'


class GmailError(RuntimeError):
    """A redacted connector failure safe to return to the browser."""


class CursorExpired(GmailError):
    """Gmail history expired; a bounded full synchronization is necessary."""


class MessageGone(GmailError):
    """A message was deleted between enumeration and retrieval."""


class PlainHTML(HTMLParser):
    """Extract inert text from email HTML; never render the original markup."""
    def __init__(self):
        """Initialize text accumulator and suppressed script/style depth."""
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        """Suppress active content and preserve common paragraph boundaries."""
        if tag in {'script', 'style'}:
            self.hidden += 1
        elif tag in {'br', 'p', 'div', 'tr'}:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        """End suppression at the matching active tag."""
        if tag in {'script', 'style'}:
            self.hidden = max(0, self.hidden - 1)

    def handle_data(self, data):
        """Collect visible text only."""
        if not self.hidden:
            self.parts.append(data)


def binding():
    """Resolve only the requested service binding, never another app's destination."""
    try:
        services = json.loads(os.environ.get('VCAP_SERVICES', '{}'))
        return next(s['credentials'] for group in services.values() for s in group if s.get('name') == SERVICE)
    except (ValueError, KeyError, StopIteration):
        raise GmailError('Gmail is not configured: bind the dedicated Destination service.') from None


class Gmail:
    """Small authenticated Gmail reader with bounded retry and MIME decoding."""
    def __init__(self):
        """Exchange destination credentials for server-only Gmail authorization."""
        creds = binding()
        try:
            uaa = creds.get('uaa', creds)
            token = requests.post(uaa['url'].rstrip('/') + '/oauth/token',
                auth=(uaa['clientid'], uaa['clientsecret']), data={'grant_type': 'client_credentials'}, timeout=30)
            token.raise_for_status()
            headers = {'Authorization': 'Bearer ' + token.json()['access_token']}
            uri = creds['uri'].rstrip('/') + '/destination-configuration/v1/'
            config = requests.get(uri + 'subaccountDestinations/' + DESTINATION, headers=headers, timeout=30)
            config.raise_for_status()
            payload = config.json()
            values = payload.get('destinationConfiguration', payload)
            if values.get('URL', '').rstrip('/') != BASE or values.get('Authentication') != 'OAuth2RefreshToken':
                raise ValueError('Invalid destination')
            properties = values.get('properties') or values.get('Properties') or {}
            additional = {p['key']: p.get('value') for p in values.get('additionalProperties', []) if 'key' in p}
            refresh = values.get('GMAIL_REFRESH_TOKEN') or properties.get('GMAIL_REFRESH_TOKEN') or additional.get('GMAIL_REFRESH_TOKEN')
            if not refresh:
                raise ValueError('Refresh token missing')
            headers['X-refresh-token'] = refresh
            resolved = requests.get(creds['uri'].rstrip('/') + '/destination-configuration/v2/destinations/' + DESTINATION + '@subaccount', headers=headers, timeout=30)
            resolved.raise_for_status()
            auth = next(a for a in resolved.json()['authTokens'] if not a.get('error'))
            self.authorization = auth.get('http_header', {}).get('value') or (auth['type'] + ' ' + auth['value'])
        except (requests.RequestException, ValueError, KeyError, StopIteration, TypeError):
            raise GmailError('Gmail authorization failed. Check the destination and renew its refresh token.') from None

    def get(self, path, **params):
        """GET Gmail JSON, retrying throttling/server failures without leaking responses."""
        for attempt in range(5):
            try:
                response = requests.get(BASE + '/users/me/' + path, params=params,
                    headers={'Authorization': self.authorization}, timeout=45)
            except requests.RequestException:
                raise GmailError('Gmail could not be reached; retry the fetch.') from None
            if response.status_code == 404 and path == 'history':
                raise CursorExpired('History cursor expired')
            if response.status_code == 404 and path.startswith('messages/'):
                raise MessageGone('Message no longer exists')
            throttled = False
            if response.status_code == 403:
                try:
                    throttled = any(e.get('reason') in {'rateLimitExceeded', 'userRateLimitExceeded'} for e in response.json().get('error', {}).get('errors', []))
                except ValueError:
                    pass
            if (throttled or response.status_code in {429, 500, 502, 503, 504}) and attempt < 4:
                retry_after = response.headers.get('Retry-After', '')
                time.sleep(min(30, int(retry_after) if retry_after.isdigit() else 2 ** attempt))
                continue
            if not response.ok:
                raise GmailError('Gmail request failed; check authorization or retry later.')
            return response.json()

    def pages(self, path, **params):
        """Yield every page, retaining the same initial cursor and query."""
        while True:
            page = self.get(path, **params)
            yield page
            if not page.get('nextPageToken'):
                return
            params['pageToken'] = page['nextPageToken']

    def message(self, message_id):
        """Decode one full MIME tree with bounded downloaded attachments and body text."""
        raw = self.get('messages/' + message_id, format='full')
        headers = {h['name'].lower(): h['value'] for h in raw.get('payload', {}).get('headers', [])}
        name, sender = parseaddr(headers.get('from', ''))
        plain, html, files, warnings = [], [], [], []
        total, seen = 0, 0

        def walk(part):
            """Visit nested MIME leaves and collect text or bounded original files."""
            nonlocal total, seen
            for child in part.get('parts', []):
                walk(child)
            body, filename = part.get('body', {}), part.get('filename', '')
            size = int(body.get('size') or 0)
            if filename:
                seen += 1
                if seen > MAX_ATTACHMENTS or size > MAX_FILE or total + size > MAX_TOTAL:
                    warnings.append({'filename': filename[:200], 'error': 'Attachment limit exceeded'})
                    return
            elif size > 1024 * 1024 or part.get('mimeType') not in {'text/plain', 'text/html'}:
                return
            encoded = body.get('data', '')
            if body.get('attachmentId'):
                encoded = self.get('messages/' + message_id + '/attachments/' + body['attachmentId']).get('data', '')
            data = base64.urlsafe_b64decode(encoded + '=' * (-len(encoded) % 4))
            if filename:
                total += len(data)
                files.append((filename, data))
            elif part.get('mimeType') == 'text/plain':
                plain.append(data.decode('utf-8', errors='replace'))
            else:
                parser = PlainHTML()
                parser.feed(data.decode('utf-8', errors='replace'))
                html.append(''.join(parser.parts))

        walk(raw.get('payload', {}))
        return {'sender': sender, 'sender_name': name, 'subject': headers.get('subject', ''),
                'body': '\n'.join(plain or html)[:1000000], 'attachments': files, 'warnings': warnings,
                'received_at': datetime.fromtimestamp(int(raw.get('internalDate', 0)) / 1000, timezone.utc).isoformat(),
                'gmail_id': message_id, 'inbox': 'INBOX' in raw.get('labelIds', [])}

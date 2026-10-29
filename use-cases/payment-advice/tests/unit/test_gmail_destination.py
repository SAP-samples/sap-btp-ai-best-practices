"""Verify Destination exchange and MIME handling without live Google credentials."""
import base64
import unittest
from unittest.mock import Mock, patch
from app.email_ingestion.gmail import Gmail, BASE, DESTINATION, GmailError


class GmailDestinationTests(unittest.TestCase):
    """Protect the server-only refresh header and explicit subaccount lookup contract."""
    def test_refresh_header_and_explicit_destination_scope(self):
        """Resolve the same subaccount destination from which the refresh token was read."""
        config = Mock()
        config.json.return_value = {'destinationConfiguration': {'URL': BASE, 'Authentication': 'OAuth2RefreshToken', 'GMAIL_REFRESH_TOKEN': 'test-refresh'}}
        resolved = Mock()
        resolved.json.return_value = {'authTokens': [{'type': 'Bearer', 'value': 'test-access'}]}
        token = Mock()
        token.json.return_value = {'access_token': 'service-test'}
        with patch('app.email_ingestion.gmail.binding', return_value={'url':'https://uaa.test','clientid':'id','clientsecret':'secret','uri':'https://destination.test'}), \
             patch('app.email_ingestion.gmail.requests.post', return_value=token), \
             patch('app.email_ingestion.gmail.requests.get', side_effect=[config, resolved]) as get:
            gmail = Gmail()
        self.assertTrue(get.call_args.args[0].endswith('/v2/destinations/' + DESTINATION + '@subaccount'))
        self.assertEqual(get.call_args.kwargs['headers']['X-refresh-token'], 'test-refresh')
        self.assertEqual(gmail.authorization, 'Bearer test-access')

    def test_mime_html_is_inert_and_nested_attachment_is_preserved(self):
        """HTML source is converted to inert text; nested attachments retain their bytes."""
        encode = lambda data: base64.urlsafe_b64encode(data).decode().rstrip('=')
        gmail = object.__new__(Gmail)
        gmail.get = Mock(return_value={'internalDate':'1000','labelIds':['INBOX'],'payload':{
            'headers':[{'name':'From','value':'Example <noreply@example.com>'}], 'parts':[
                {'mimeType':'text/html','body':{'data':encode(b'<p>Hello</p><script>alert(1)</script>')}},
                {'mimeType':'multipart/mixed','parts':[{'filename':'test.txt','body':{'size':4,'data':encode(b'test')}}]}]}})
        message = gmail.message('test-id')
        self.assertEqual(message['body'].strip(), 'Hello')
        self.assertEqual(message['attachments'], [('test.txt', b'test')])


if __name__ == '__main__':
    unittest.main()

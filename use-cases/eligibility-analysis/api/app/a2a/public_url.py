"""Resolve public A2A discovery endpoints without advertising localhost in Cloud Foundry."""
import json
from urllib.parse import urlparse


def public_agent_endpoint(environ):
    """Prefer an explicit endpoint/origin, then CF route, with localhost only in development."""
    endpoint=environ.get('A2A_ENDPOINT_URL')
    if not endpoint:
        origin=environ.get('AGENT_PUBLIC_URL') or environ.get('A2A_BASE_URL') or environ.get('API_BASE_URL')
        if not origin and environ.get('VCAP_APPLICATION'):
            routes=json.loads(environ['VCAP_APPLICATION']).get('application_uris',[])
            if routes:origin='https://'+routes[0]
        if not origin:
            if environ.get('VCAP_APPLICATION') or environ.get('APP_ENV')=='production':raise ValueError('Configure a public A2A route')
            origin='http://localhost:8000'
        endpoint=origin.rstrip('/')
        if not endpoint.endswith('/api/a2a'):endpoint+='/api/a2a'
    endpoint=endpoint.rstrip('/');parsed=urlparse(endpoint)
    if parsed.scheme not in ('https','http') or not parsed.hostname:raise ValueError('A2A endpoint must be an HTTP(S) URL')
    if (environ.get('VCAP_APPLICATION') or environ.get('APP_ENV')=='production') and parsed.hostname in ('localhost','127.0.0.1','0.0.0.0'):
        raise ValueError('Production discovery cannot advertise localhost')
    return endpoint

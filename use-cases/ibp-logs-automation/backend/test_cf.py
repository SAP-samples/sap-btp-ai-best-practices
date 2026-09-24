import json
import os
from playwright.sync_api import sync_playwright

with open('/app/session_cookies.json') as f:
    data = json.load(f)
raw = data.get('cookies', data) if isinstance(data, dict) else data
pw_cookies = []
for c in raw:
    nc = {'name': c['name'], 'value': c['value'], 'domain': c['domain'],
          'path': c.get('path','/'), 'secure': c.get('secure',False),
          'httpOnly': c.get('httpOnly',False),
          'sameSite': c.get('sameSite','Lax') if c.get('sameSite') in ('Strict','Lax','None') else 'Lax'}
    if 'expirationDate' in c:
        nc['expires'] = int(c['expirationDate'])
    pw_cookies.append(nc)

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, args=['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage'])
    ctx = b.new_context(ignore_https_errors=True)
    ctx.add_cookies(pw_cookies)
    p = ctx.new_page()
    js_ok = p.evaluate('() => 1+1')
    print('JS basic:', js_ok)
    ibp_url = os.getenv('IBP_URL', 'https://<your-ibp-tenant>.scmibp.ondemand.com/')
    p.goto(ibp_url, wait_until='domcontentloaded', timeout=30000)
    p.wait_for_timeout(8000)
    title = p.evaluate('() => document.title')
    scripts = p.evaluate('() => document.scripts.length')
    shell = p.evaluate('() => document.querySelectorAll("[class*=Shell],[class*=shell],[id*=shell]").length')
    print('Title:', title)
    print('Scripts:', scripts)
    print('Shell elements:', shell)
    p.screenshot(path='/app/downloads/debug2.png')
    b.close()
    print('Done')

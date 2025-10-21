import os, json, time, argparse, asyncio
from dotenv import load_dotenv

from botbuilder.core import BotFrameworkAdapter, BotFrameworkAdapterSettings, TurnContext
from botbuilder.schema import ConversationReference, Activity, ActivityTypes
import requests

parser = argparse.ArgumentParser(description="Send a proactive Teams message or forward alert to app.py after a delay.")
parser.add_argument("--delay", type=int, default=30, help="Delay in seconds before sending.")

# --- direct-to-Teams mode (legacy) ---
parser.add_argument("--ref", default="conversation_ref.json", help="Path to conversation reference JSON.")
parser.add_argument("--conversation-id", default=None, help="Conversation id (if JSON is a dict).")
parser.add_argument("--text", default="👋 Proactive ping from server.", help="Plain text if not using card.")

# --- invoice card (direct mode) ---
parser.add_argument("--invoice", default=None, help="Invoice number: send an Adaptive Card with Validate button.")
parser.add_argument("--title", default="Invoice Alert", help="Card title when --invoice is set.")

# --- app intake mode ---
parser.add_argument("--app-url", default=None, help="If set (e.g. http://localhost:3978/alerts), post alert to the bot app instead of Teams.")
parser.add_argument("--type", default="generic", choices=["generic","not_received","po_mismatch"], help="Issue type for /alerts.")
parser.add_argument("--po", dest="po_number", default=None, help="PO number for /alerts.")
parser.add_argument("--recipient", default="Marta", help="Recipient display name for message text.")

args = parser.parse_args()

# -- if app-url is provided, use intake mode --
if args.app_url:
    payload = {
        "type": args.type,
        "po_number": args.po_number,
        "invoice_number": args.invoice,
        "recipient_name": args.recipient,
        # conversation_id is optional; app will use last known if omitted
    }
    print(f"[INIT] will POST to {args.app_url} in {args.delay} seconds…")
    time.sleep(args.delay)
    r = requests.post(args.app_url, json=payload, timeout=30)
    print(f"[DONE] app response: {r.status_code} {r.text[:300]}")
    raise SystemExit(0)

# --- otherwise: direct-to-Teams mode (as before) ---
load_dotenv()
APP_ID = os.getenv("MICROSOFT_APP_ID")
APP_PWD = os.getenv("MICROSOFT_APP_PASSWORD")
if not APP_ID or not APP_PWD:
    raise RuntimeError("MICROSOFT_APP_ID / MICROSOFT_APP_PASSWORD are not set in .env")

if not os.path.exists(args.ref):
    raise FileNotFoundError(f"Reference file not found: {args.ref}")

with open(args.ref, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict) and "serviceUrl" in data:
    ref_data = data
    chosen_conv_id = ref_data.get("conversation", {}).get("id", "unknown")
else:
    if not isinstance(data, dict) or len(data) == 0:
        raise ValueError("Invalid reference JSON format.")
    if args.conversation_id and args.conversation_id in data:
        ref_data = data[args.conversation_id]
        chosen_conv_id = args.conversation_id
    else:
        chosen_conv_id, ref_data = list(data.items())[-1]

reference = ConversationReference().deserialize(ref_data)
service_url = reference.service_url

try:
    from botframework.connector.auth import MicrosoftAppCredentials
    MicrosoftAppCredentials.trust_service_url(service_url)
except Exception:
    pass

adapter = BotFrameworkAdapter(BotFrameworkAdapterSettings(APP_ID, APP_PWD))

def build_invoice_card(invoice: str, title: str):
    return {
        "type": "AdaptiveCard",
        "version": "1.4",
        "body": [
            {"type": "TextBlock", "text": f"📣 {title}", "size": "Large", "weight": "Bolder"},
            {"type": "TextBlock",
             "text": f"There are open issues for invoice **{invoice}**.",
             "wrap": True}
        ],
        "actions": [
            {"type": "Action.Submit", "title": "Validate now",
             "data": {"action": "validate_direct", "invoice_number": str(invoice)}}
        ],
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json"
    }

async def send_once():
    async def logic(turn_context: TurnContext):
        if args.invoice:
            card = build_invoice_card(args.invoice, args.title)
            att = {"contentType":"application/vnd.microsoft.card.adaptive", "content": card}
            await turn_context.send_activity(Activity(type=ActivityTypes.message, attachments=[att]))
            print(f"[PROACTIVE] card sent (invoice={args.invoice}) -> conversation={chosen_conv_id}")
        else:
            await turn_context.send_activity(args.text)
            print(f"[PROACTIVE] text sent -> conversation={chosen_conv_id}")
    await adapter.continue_conversation(reference, logic, APP_ID)

print(f"[INIT] using conversation_id={chosen_conv_id}")
print(f"[INIT] serviceUrl={service_url}")
print(f"[INIT] will send in {args.delay} seconds…")
time.sleep(args.delay)
asyncio.run(send_once())
print("[DONE]")

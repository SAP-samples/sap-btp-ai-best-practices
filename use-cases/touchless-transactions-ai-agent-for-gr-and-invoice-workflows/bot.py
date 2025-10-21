# bot.py
import logging
from typing import List
from botbuilder.core import TurnContext, CardFactory
from botbuilder.core.teams import TeamsActivityHandler
from botbuilder.schema import Activity, Attachment, ChannelAccount

from cards import build_gr_confirmation_card

log = logging.getLogger("bot")

class TeamsGrBot(TeamsActivityHandler):
    """
    Бот, который отправляет таблицу GR:
    - при добавлении в чат (membersAdded)
    - по сообщению 'gr'
    """

    def __init__(self, adapter, app_id: str):
        super().__init__()
        self.adapter = adapter
        self.app_id = app_id

    # ----- helpers -----
    def _demo_rows(self) -> List[dict]:
        # Синтетические данные, как в ваших скриншотах
        return [
            {"line": 15, "invoice": "5109058689", "po": "4501366628", "ir_amount": 1000, "po_amount": 1000, "gr_current": 0,     "gr_confirm": 1000},
            {"line": 16, "invoice": "5109058689", "po": "—",        "ir_amount": 1000, "po_amount": 1000, "gr_current": 500,   "gr_confirm": 500},
            {"line": 17, "invoice": "5109058689", "po": "—",        "ir_amount": 1005, "po_amount": 1000, "gr_current": 0,     "gr_confirm": 0},
            {"line": 18, "invoice": "5109058689", "po": "—",        "ir_amount": 1005, "po_amount": 1000, "gr_current": 500,   "gr_confirm": 505},
            {"line": 19, "invoice": "5109058689", "po": "—",        "ir_amount": 1249, "po_amount": 1000, "gr_current": 0,     "gr_confirm": 1249},
            {"line": 20, "invoice": "5109058689", "po": "—",        "ir_amount": 1249, "po_amount": 1000, "gr_current": 0,     "gr_confirm": 749},
            {"line": 21, "invoice": "5109058689", "po": "—",        "ir_amount": 100251, "po_amount": 100000, "gr_current": 0, "gr_confirm": 100251},
            {"line": 22, "invoice": "5109058689", "po": "—",        "ir_amount": 100251, "po_amount": 100000, "gr_current": 50000, "gr_confirm": 50251},
        ]

    async def _send_gr_card(self, turn_context: TurnContext):
        rows = self._demo_rows()
        card_dict = build_gr_confirmation_card(
            invoice_no="5109058689",
            items=rows,
            use_table=True,  # если в окружении вдруг нет Table 1.5 -> False (будет ColumnSet)
        )
        attachment: Attachment = CardFactory.adaptive_card(card_dict)
        await turn_context.send_activity(Activity(type="message", attachments=[attachment]))

    # ----- events -----
    async def on_message_activity(self, turn_context: TurnContext):
        text = (turn_context.activity.text or "").strip().lower()
        log.info(
            "message: text=%r channel=%s convId=%s serviceUrl=%s",
            text,
            turn_context.activity.channel_id,
            turn_context.activity.conversation.id if turn_context.activity.conversation else None,
            turn_context.activity.service_url,
        )
        if text in ("gr", "confirm", "table", "demo"):
            await self._send_gr_card(turn_context)
        else:
            await turn_context.send_activity(
                "🤖 Commands: `gr` — show GR confirmation table."
            )

    async def on_members_added_activity(
        self, members_added: List[ChannelAccount], turn_context: TurnContext
    ):
        ids = [m.id for m in members_added or []]
        log.info(
            "membersAdded: channel=%s convId=%s serviceUrl=%s members=%s",
            turn_context.activity.channel_id,
            turn_context.activity.conversation.id if turn_context.activity.conversation else None,
            turn_context.activity.service_url,
            ids,
        )
        # Приветствие + сразу карточка
        await turn_context.send_activity("Hi! I'll show the GR confirmation table.")
        await self._send_gr_card(turn_context)
